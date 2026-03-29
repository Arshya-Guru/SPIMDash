"""GPU-native SynthSeg synthetic image generator — Experiment B.

Changes from gpu_synth.py:
1. Affine augmentation (rotation +-8deg, scaling 0.9-1.1x) before elastic
2. Probabilistic augmentations (elastic 50%, bias 80%, gamma 50%)
3. Background-aware intensity (bg=low, brain=high)
4. Mandatory boundary smoothing (sigma 0.5-2.0, always on)
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
    return torch.from_numpy(out).unsqueeze(0).unsqueeze(0)


class GPUSynthGeneratorV2:
    """GPU-native SynthSeg generator with SPIM-aware probabilistic augmentation."""

    def __init__(self, label_maps_np: list[np.ndarray], n_labels: int,
                 target_shape: tuple[int, int, int], device: torch.device,
                 cfg: dict = None):
        self.n_labels = n_labels
        self.target_shape = target_shape
        self.device = device
        cfg = cfg or {}
        self.cfg = cfg

        olf_labels = cfg.get("olfactory_labels", [1, 2])

        print(f"[GPU Synth V2] Loading {len(label_maps_np)} label maps to {device}...")
        print(f"[GPU Synth V2] Merging olfactory labels {olf_labels} -> 0 (background)")
        self.label_maps = []
        for lm_np in label_maps_np:
            lm = lm_np.copy()
            for ol in olf_labels:
                lm[lm == ol] = 0
            t = conform_and_to_tensor(lm, target_shape).to(device)
            self.label_maps.append(t)
        mem_gb = len(self.label_maps) * self.label_maps[0].nelement() * 4 / 1e9
        print(f"[GPU Synth V2] {len(self.label_maps)} maps on GPU, ~{mem_gb:.1f} GB")

        D, H, W = target_shape

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

    def _random_affine_grid(self) -> torch.Tensor:
        """Generate a random affine-transformed grid (rotation + scaling)."""
        device = self.device
        D, H, W = self.target_shape

        # Random rotation: +-8 degrees per axis
        angles = (torch.rand(3, device=device) * 2 - 1) * (8.0 * 3.14159 / 180.0)
        cx, cy, cz = torch.cos(angles)
        sx, sy, sz = torch.sin(angles)

        Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], device=device, dtype=torch.float32)
        Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], device=device, dtype=torch.float32)
        Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], device=device, dtype=torch.float32)
        R = Rz @ Ry @ Rx

        # Random scaling: 0.9-1.1 per axis
        s = torch.rand(3, device=device) * 0.2 + 0.9
        S = torch.diag(s)

        A = R @ S
        theta = torch.zeros(1, 3, 4, device=device)
        theta[0, :3, :3] = A

        return F.affine_grid(theta, [1, 1, D, H, W], align_corners=False)

    @torch.no_grad()
    def generate(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate batch of (images, labels) on GPU with probabilistic augmentation."""
        images = []
        labels = []

        for _ in range(batch_size):
            idx = torch.randint(len(self.label_maps), (1,)).item()
            label_vol = self.label_maps[idx]

            # Spatial: always affine, 50% elastic on top
            if torch.rand(1, device=self.device).item() < 0.5:
                label_vol = self._affine_elastic_deform(label_vol)
            else:
                # Affine only (no elastic)
                grid = self._random_affine_grid()
                label_vol = F.grid_sample(label_vol, grid, mode='nearest',
                                          padding_mode='border', align_corners=False)

            label_long = label_vol.squeeze(0).squeeze(0).long()

            # Intensity (background-aware)
            image = self._sample_intensities(label_long)

            # Mandatory boundary smoothing (sigma 0.5-2.0)
            image = self._smooth_boundaries(image)

            # Bias field (80% probability)
            if torch.rand(1, device=self.device).item() < 0.8:
                image = self._add_bias_field(image)

            # Noise (always)
            image = self._add_noise(image)

            # Blur / resolution variation (always)
            image = self._add_blur(image)

            # Gamma (50% probability)
            if torch.rand(1, device=self.device).item() < 0.5:
                image = self._gamma_augment(image)

            # Normalize to [0, 1]
            vmin, vmax = image.min(), image.max()
            image = (image - vmin) / (vmax - vmin + 1e-8)

            images.append(image.unsqueeze(0))
            labels.append(label_long)

        return torch.stack(images), torch.stack(labels)

    def _affine_elastic_deform(self, label_vol: torch.Tensor) -> torch.Tensor:
        """Affine + elastic deformation combined."""
        scale = self.cfg.get("deform_scale", 3.0)
        D, H, W = self.target_shape

        # Start from affine-augmented grid
        base = self._random_affine_grid()

        # Add elastic displacement
        flow = torch.randn(1, 3, *self.flow_shape, device=self.device) * scale
        flow = F.interpolate(flow, size=(D, H, W), mode='trilinear', align_corners=False)

        flow_grid = flow.permute(0, 2, 3, 4, 1)
        flow_grid = flow_grid[..., [2, 1, 0]]
        flow_grid = flow_grid * self.norm_factor

        grid = base + flow_grid

        return F.grid_sample(
            label_vol, grid, mode='nearest',
            padding_mode='border', align_corners=False,
        )

    def _sample_intensities(self, label_map: torch.Tensor) -> torch.Tensor:
        """Background-aware intensity: bg=low, brain=high."""
        sig_lo, sig_hi = self.cfg.get("intensity_sigma_range", [5, 30])

        # Background gets low intensity
        bg_mean = torch.empty(1, device=self.device).uniform_(0, 50)
        bg_std = torch.empty(1, device=self.device).uniform_(1, 10)

        # Brain regions get higher intensity
        means = torch.empty(self.n_labels, device=self.device).uniform_(50, 300)
        stds = torch.empty(self.n_labels, device=self.device).uniform_(sig_lo, sig_hi)

        # Override background
        means[0] = bg_mean.item()
        stds[0] = bg_std.item()

        mean_map = means[label_map]
        std_map = stds[label_map]
        return torch.normal(mean_map, std_map)

    def _smooth_boundaries(self, image: torch.Tensor) -> torch.Tensor:
        """Mandatory smoothing to soften hard label boundaries. Sigma 0.5-2.0."""
        sigma = torch.empty(1, device=self.device).uniform_(0.5, 2.0).item()

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

    def _add_bias_field(self, image: torch.Tensor) -> torch.Tensor:
        """Smooth multiplicative bias field."""
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
        """Optional additional blur for resolution variation."""
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
