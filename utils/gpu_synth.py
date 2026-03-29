"""GPU-native SynthSeg synthetic image generator. All ops on CUDA."""

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


class GPUSynthGenerator:
    """
    GPU-native SynthSeg-style synthetic image generator.

    All label maps stored on GPU. Each generate() call produces a fresh
    (synthetic_image, label_map) pair entirely on GPU in ~20-50ms.
    No DataLoader, no CPU workers, no scipy.
    """

    def __init__(self, label_maps_np: list[np.ndarray], n_labels: int,
                 target_shape: tuple[int, int, int], device: torch.device,
                 cfg: dict = None):
        self.n_labels = n_labels
        self.target_shape = target_shape
        self.device = device
        cfg = cfg or {}
        self.cfg = cfg

        # Merge olfactory bulb labels to background before loading
        # ABAv3 roi-22: labels 21,22 are left/right olfactory areas
        olf_labels = cfg.get("olfactory_labels", [1, 2])

        print(f"[GPU Synth] Loading {len(label_maps_np)} label maps to {device}...")
        print(f"[GPU Synth] Merging olfactory labels {olf_labels} -> 0 (background)")
        self.label_maps = []
        for lm_np in label_maps_np:
            lm = lm_np.copy()
            for ol in olf_labels:
                lm[lm == ol] = 0
            t = conform_and_to_tensor(lm, target_shape).to(device)  # (1,1,D,H,W)
            self.label_maps.append(t)
        mem_gb = len(self.label_maps) * self.label_maps[0].nelement() * 4 / 1e9
        print(f"[GPU Synth] {len(self.label_maps)} maps on GPU, ~{mem_gb:.1f} GB")

        D, H, W = target_shape

        # Pre-compute base grid for grid_sample (normalized [-1, 1] coords)
        self.base_grid = F.affine_grid(
            torch.eye(3, 4, device=device).unsqueeze(0),
            [1, 1, D, H, W],
            align_corners=False,
        )  # (1, D, H, W, 3)

        # Coarse displacement grid shape
        spacing = cfg.get("deform_grid_spacing", 32)
        self.flow_shape = (max(1, D // spacing), max(1, H // spacing), max(1, W // spacing))

        # Normalization: convert voxel displacements to [-1,1] range for grid_sample
        # grid_sample grid last dim is (x, y, z) = (W, H, D) order
        self.norm_factor = torch.tensor(
            [2.0 / W, 2.0 / H, 2.0 / D],
            device=device,
        ).reshape(1, 1, 1, 1, 3)

    @torch.no_grad()
    def generate(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate batch of (images, labels) entirely on GPU. ~30-50ms."""
        images = []
        labels = []

        for _ in range(batch_size):
            # 1. Random label map from GPU pool
            idx = torch.randint(len(self.label_maps), (1,)).item()
            label_vol = self.label_maps[idx]  # (1, 1, D, H, W) float32

            # 2. Elastic deformation via grid_sample (nearest-neighbor)
            label_vol = self._elastic_deform(label_vol)
            label_long = label_vol.squeeze(0).squeeze(0).long()  # (D, H, W)

            # 3. Intensity sampling
            image = self._sample_intensities(label_long)

            # 4. Bias field
            image = self._add_bias_field(image)

            # 5. Noise
            image = self._add_noise(image)

            # 6. Blur
            image = self._add_blur(image)

            # 7. Gamma
            image = self._gamma_augment(image)

            # 8. Normalize to [0, 1]
            vmin, vmax = image.min(), image.max()
            image = (image - vmin) / (vmax - vmin + 1e-8)

            images.append(image.unsqueeze(0))  # (1, D, H, W)
            labels.append(label_long)

        return torch.stack(images), torch.stack(labels)

    def _elastic_deform(self, label_vol: torch.Tensor) -> torch.Tensor:
        """GPU elastic deformation using grid_sample with mode='nearest'."""
        scale = self.cfg.get("deform_scale", 3.0)
        D, H, W = self.target_shape

        # Small random displacement: (1, 3, fd, fh, fw)
        flow = torch.randn(1, 3, *self.flow_shape, device=self.device) * scale

        # Upsample to full volume resolution
        flow = F.interpolate(flow, size=(D, H, W), mode='trilinear', align_corners=False)

        # Convert to grid_sample format:
        # permute (1,3,D,H,W) -> (1,D,H,W,3)
        # reorder last dim from (D,H,W) -> (W,H,D) = (x,y,z) for grid_sample
        # scale voxel displacements to [-1,1] range
        flow_grid = flow.permute(0, 2, 3, 4, 1)   # (1, D, H, W, 3)
        flow_grid = flow_grid[..., [2, 1, 0]]       # D,H,W -> W,H,D = x,y,z
        flow_grid = flow_grid * self.norm_factor     # voxels -> normalized

        grid = self.base_grid + flow_grid

        deformed = F.grid_sample(
            label_vol, grid, mode='nearest',
            padding_mode='border', align_corners=False,
        )
        return deformed

    def _sample_intensities(self, label_map: torch.Tensor) -> torch.Tensor:
        """Vectorized GPU intensity: LUT of per-label mean/std, sample normal."""
        mu_lo, mu_hi = self.cfg.get("intensity_mu_range", [0, 300])
        sig_lo, sig_hi = self.cfg.get("intensity_sigma_range", [5, 30])

        means = torch.empty(self.n_labels, device=self.device).uniform_(mu_lo, mu_hi)
        stds = torch.empty(self.n_labels, device=self.device).uniform_(sig_lo, sig_hi)

        mean_map = means[label_map]
        std_map = stds[label_map]
        return torch.normal(mean_map, std_map)

    def _add_bias_field(self, image: torch.Tensor) -> torch.Tensor:
        """Smooth multiplicative bias via trilinear upsampling of small random tensor."""
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
        """Separable 3D Gaussian blur via three 1D convolutions."""
        lo, hi = self.cfg.get("blur_sigma_range", [0, 1.5])
        sigma = torch.empty(1, device=self.device).uniform_(lo, hi).item()
        if sigma < 0.1:
            return image

        k_size = int(4 * sigma + 0.5) * 2 + 1
        k_size = max(k_size, 3)
        x = torch.arange(k_size, device=self.device, dtype=torch.float32) - k_size // 2
        k1d = torch.exp(-0.5 * x ** 2 / (sigma ** 2 + 1e-8))
        k1d = k1d / k1d.sum()

        img = image.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        pad_d = (k_size // 2, 0, 0)
        pad_h = (0, k_size // 2, 0)
        pad_w = (0, 0, k_size // 2)

        img = F.conv3d(img, k1d.reshape(1, 1, -1, 1, 1), padding=pad_d)
        img = F.conv3d(img, k1d.reshape(1, 1, 1, -1, 1), padding=pad_h)
        img = F.conv3d(img, k1d.reshape(1, 1, 1, 1, -1), padding=pad_w)
        return img.squeeze(0).squeeze(0)

    def _gamma_augment(self, image: torch.Tensor) -> torch.Tensor:
        lo, hi = self.cfg.get("gamma_range", [0.7, 1.5])
        gamma = torch.empty(1, device=self.device).uniform_(lo, hi).item()
        mn = image.min()
        image = image - mn
        mx = image.max() + 1e-8
        image = torch.pow(image / mx, gamma) * mx
        return image + mn
