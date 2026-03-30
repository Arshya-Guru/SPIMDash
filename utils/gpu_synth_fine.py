"""GPU-native SynthSeg generator using fine parcellations for intensity sampling.

Loads TWO label maps per subject (pixel-aligned, same shape):
  - Fine label map (roi198 or roi-all) → used for intensity painting (each sub-region
    gets its own random intensity, producing realistic intra-region texture)
  - Coarse label map (roi22) → used directly as the training target

The model still predicts 22 classes. Only the synthetic image appearance changes.
"""

import torch
import torch.nn.functional as F
import numpy as np


def conform_and_to_tensor(vol_np: np.ndarray, target_shape: tuple) -> torch.Tensor:
    """Pad/crop numpy volume to target_shape, return as (1, 1, D, H, W) float32 tensor."""
    D, H, W = target_shape
    out = np.zeros((D, H, W), dtype=np.float32)
    sd, sh, sw = vol_np.shape[:3]
    d0, h0, w0 = max(0, (sd - D) // 2), max(0, (sh - H) // 2), max(0, (sw - W) // 2)
    od, oh, ow = max(0, (D - sd) // 2), max(0, (H - sh) // 2), max(0, (W - sw) // 2)
    cd, ch, cw = min(D, sd), min(H, sh), min(W, sw)
    out[od:od+cd, oh:oh+ch, ow:ow+cw] = vol_np[d0:d0+cd, h0:h0+ch, w0:w0+cw]
    return torch.from_numpy(out).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)


class GPUSynthGeneratorFine:
    """GPU-native SynthSeg generator with fine parcellation for intensity sampling.

    Stores two label maps per subject on GPU:
      - fine_maps: high-resolution parcellation for intensity painting
      - coarse_maps: roi22 parcellation for segmentation target
    """

    def __init__(
        self,
        fine_labels_np: list[np.ndarray],
        coarse_labels_np: list[np.ndarray],
        n_labels: int,
        target_shape: tuple[int, int, int],
        device: torch.device,
        cfg: dict = None,
    ):
        assert len(fine_labels_np) == len(coarse_labels_np)
        self.n_labels = n_labels
        self.target_shape = target_shape
        self.device = device
        cfg = cfg or {}
        self.cfg = cfg

        olf_labels = cfg.get("olfactory_labels", [1, 2])

        # Count unique fine labels for intensity sampling
        all_fine = set()
        for f in fine_labels_np:
            all_fine.update(np.unique(f).tolist())
        self.n_fine_labels = max(all_fine) + 1
        print(f"[GPU Synth Fine] {len(all_fine)} unique fine labels (max index {max(all_fine)})")

        print(f"[GPU Synth Fine] Loading {len(fine_labels_np)} subject pairs to {device}...")
        print(f"[GPU Synth Fine] Merging olfactory labels {olf_labels} -> 0 in coarse targets")

        self.fine_maps = []
        self.coarse_maps = []
        for fine_np, coarse_np in zip(fine_labels_np, coarse_labels_np):
            # Fine map: no merging — used only for intensity painting
            ft = conform_and_to_tensor(fine_np, target_shape).to(device)
            self.fine_maps.append(ft)

            # Coarse map: merge olfactory to background (same as Experiment A)
            cm = coarse_np.copy()
            for ol in olf_labels:
                cm[cm == ol] = 0
            ct = conform_and_to_tensor(cm, target_shape).to(device)
            self.coarse_maps.append(ct)

        mem_gb = (len(self.fine_maps) + len(self.coarse_maps)) * self.fine_maps[0].nelement() * 4 / 1e9
        print(f"[GPU Synth Fine] {len(self.fine_maps)} pairs on GPU, ~{mem_gb:.1f} GB")

        D, H, W = target_shape

        # Pre-compute base grid for grid_sample
        self.base_grid = F.affine_grid(
            torch.eye(3, 4, device=device).unsqueeze(0),
            [1, 1, D, H, W],
            align_corners=False,
        )

        spacing = cfg.get("deform_grid_spacing", 32)
        self.flow_shape = (max(1, D // spacing), max(1, H // spacing), max(1, W // spacing))

        self.norm_factor = torch.tensor(
            [2.0 / W, 2.0 / H, 2.0 / D],
            device=device,
        ).reshape(1, 1, 1, 1, 3)

    @torch.no_grad()
    def generate(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate batch of (images, labels) on GPU.

        Images are painted using fine parcellation intensities.
        Labels are the coarse roi22 targets.
        """
        images = []
        labels = []

        for _ in range(batch_size):
            idx = torch.randint(len(self.fine_maps), (1,)).item()
            fine_vol = self.fine_maps[idx]    # (1, 1, D, H, W)
            coarse_vol = self.coarse_maps[idx]  # (1, 1, D, H, W)

            # Apply SAME deformation to both maps (shared grid)
            grid = self._make_deform_grid()
            fine_def = F.grid_sample(fine_vol, grid, mode='nearest',
                                     padding_mode='border', align_corners=False)
            coarse_def = F.grid_sample(coarse_vol, grid, mode='nearest',
                                       padding_mode='border', align_corners=False)

            fine_long = fine_def.squeeze(0).squeeze(0).long()
            coarse_long = coarse_def.squeeze(0).squeeze(0).long()

            # Intensity sampling from FINE labels
            image = self._sample_intensities(fine_long)

            # Augmentations (identical to gpu_synth.py)
            image = self._add_bias_field(image)
            image = self._add_noise(image)
            image = self._add_blur(image)
            image = self._gamma_augment(image)

            # Normalize to [0, 1]
            vmin, vmax = image.min(), image.max()
            image = (image - vmin) / (vmax - vmin + 1e-8)

            images.append(image.unsqueeze(0))
            labels.append(coarse_long)  # target is roi22

        return torch.stack(images), torch.stack(labels)

    def _make_deform_grid(self) -> torch.Tensor:
        """Generate elastic deformation grid (same logic as gpu_synth.py)."""
        scale = self.cfg.get("deform_scale", 3.0)
        D, H, W = self.target_shape

        flow = torch.randn(1, 3, *self.flow_shape, device=self.device) * scale
        flow = F.interpolate(flow, size=(D, H, W), mode='trilinear', align_corners=False)

        flow_grid = flow.permute(0, 2, 3, 4, 1)
        flow_grid = flow_grid[..., [2, 1, 0]]
        flow_grid = flow_grid * self.norm_factor

        return self.base_grid + flow_grid

    def _sample_intensities(self, label_map: torch.Tensor) -> torch.Tensor:
        """Sample random intensity per FINE label (many more regions = richer texture)."""
        mu_lo, mu_hi = self.cfg.get("intensity_mu_range", [0, 300])
        sig_lo, sig_hi = self.cfg.get("intensity_sigma_range", [5, 30])

        means = torch.empty(self.n_fine_labels, device=self.device).uniform_(mu_lo, mu_hi)
        stds = torch.empty(self.n_fine_labels, device=self.device).uniform_(sig_lo, sig_hi)

        mean_map = means[label_map]
        std_map = stds[label_map]
        return torch.normal(mean_map, std_map)

    def _add_bias_field(self, image: torch.Tensor) -> torch.Tensor:
        n = self.cfg.get("bias_field_coeffs", 4)
        strength = self.cfg.get("bias_field_strength", 0.4)
        D, H, W = self.target_shape
        small = torch.randn(1, 1, n, n, n, device=self.device)
        bias = F.interpolate(small, size=(D, H, W), mode='trilinear', align_corners=False)
        bias = bias.squeeze(0).squeeze(0)
        bias = 1.0 + strength * (bias / (bias.abs().max() + 1e-8))
        return image * bias

    def _add_noise(self, image: torch.Tensor) -> torch.Tensor:
        lo, hi = self.cfg.get("noise_std_range", [0, 25])
        std = torch.empty(1, device=self.device).uniform_(lo, hi).item()
        if std > 0.1:
            image = image + torch.randn_like(image) * std
        return image

    def _add_blur(self, image: torch.Tensor) -> torch.Tensor:
        lo, hi = self.cfg.get("blur_sigma_range", [0, 1.5])
        sigma = torch.empty(1, device=self.device).uniform_(lo, hi).item()
        if sigma < 0.1:
            return image
        k_size = int(4 * sigma + 0.5) * 2 + 1
        k_size = max(k_size, 3)
        x = torch.arange(k_size, device=self.device, dtype=torch.float32) - k_size // 2
        k1d = torch.exp(-0.5 * x ** 2 / (sigma ** 2 + 1e-8))
        k1d = k1d / k1d.sum()
        img = image.unsqueeze(0).unsqueeze(0)
        pad = k_size // 2
        img = F.conv3d(img, k1d.reshape(1, 1, -1, 1, 1), padding=(pad, 0, 0))
        img = F.conv3d(img, k1d.reshape(1, 1, 1, -1, 1), padding=(0, pad, 0))
        img = F.conv3d(img, k1d.reshape(1, 1, 1, 1, -1), padding=(0, 0, pad))
        return img.squeeze(0).squeeze(0)

    def _gamma_augment(self, image: torch.Tensor) -> torch.Tensor:
        lo, hi = self.cfg.get("gamma_range", [0.7, 1.5])
        gamma = torch.empty(1, device=self.device).uniform_(lo, hi).item()
        mn = image.min()
        image = image - mn
        mx = image.max() + 1e-8
        image = torch.pow(image / mx, gamma) * mx
        return image + mn
