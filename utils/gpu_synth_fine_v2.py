"""GPU-native SynthSeg generator with fine parcellations + full SynthSeg-faithful augmentation.

Compared to gpu_synth_fine.py, adds:
1. Random affine transforms (rotation +-15deg, scaling 0.8-1.2x)
2. Left-right flipping with bilateral label swapping (50%)
3. Background-aware intensity (30% black, 40% dim, 30% standard)
4. Gamma augmentation (always applied, gamma = exp(N(0, 0.5)))
5. Wider intensity std range [0, 30]
6. No separate additive noise step (per-voxel sampling is sufficient)
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


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


# Bilateral label pairs in roi22 (odd=left, even=right, mirrored on dim0).
# Labels 1,2 (olfactory) already merged to 0, so only 3-22 matter.
BILATERAL_PAIRS = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12),
                   (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]
# Flip axis: dim0 (the left-right axis, verified empirically)
FLIP_AXIS = 0  # in (D, H, W), D is left-right


class GPUSynthGeneratorFineV2:
    """GPU-native SynthSeg generator with fine parcellation + full augmentation suite."""

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

        # Discover max fine label index for intensity LUT sizing
        all_fine = set()
        for f in fine_labels_np:
            all_fine.update(np.unique(f).tolist())
        self.n_fine_labels = max(all_fine) + 1
        print(f"[GPU Synth FineV2] {len(all_fine)} unique fine labels (max index {max(all_fine)})")

        print(f"[GPU Synth FineV2] Loading {len(fine_labels_np)} subject pairs to {device}...")
        print(f"[GPU Synth FineV2] Merging olfactory labels {olf_labels} -> 0 in coarse targets")

        self.fine_maps = []
        self.coarse_maps = []
        for fine_np, coarse_np in zip(fine_labels_np, coarse_labels_np):
            ft = conform_and_to_tensor(fine_np, target_shape).to(device)
            self.fine_maps.append(ft)

            cm = coarse_np.copy()
            for ol in olf_labels:
                cm[cm == ol] = 0
            ct = conform_and_to_tensor(cm, target_shape).to(device)
            self.coarse_maps.append(ct)

        mem_gb = (len(self.fine_maps) + len(self.coarse_maps)) * self.fine_maps[0].nelement() * 4 / 1e9
        print(f"[GPU Synth FineV2] {len(self.fine_maps)} pairs on GPU, ~{mem_gb:.1f} GB")

        D, H, W = target_shape

        # Pre-compute identity grid
        self.base_grid = F.affine_grid(
            torch.eye(3, 4, device=device).unsqueeze(0),
            [1, 1, D, H, W],
            align_corners=False,
        )

        spacing = cfg.get("deform_grid_spacing", 32)
        self.flow_shape = (max(1, D // spacing), max(1, H // spacing), max(1, W // spacing))

        # grid_sample normalization: voxel displacement -> [-1,1]
        # grid_sample last dim = (x, y, z) = (W, H, D)
        self.norm_factor = torch.tensor(
            [2.0 / W, 2.0 / H, 2.0 / D],
            device=device,
        ).reshape(1, 1, 1, 1, 3)

        # Build bilateral swap LUT on GPU (for LR flipping)
        swap_lut = torch.arange(n_labels, device=device, dtype=torch.long)
        for l, r in BILATERAL_PAIRS:
            if l < n_labels and r < n_labels:
                swap_lut[l] = r
                swap_lut[r] = l
        self.swap_lut = swap_lut

    @torch.no_grad()
    def generate(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate batch of (images, labels) with full SynthSeg augmentation."""
        images = []
        labels = []

        for _ in range(batch_size):
            idx = torch.randint(len(self.fine_maps), (1,)).item()
            fine_vol = self.fine_maps[idx]
            coarse_vol = self.coarse_maps[idx]

            # 1. Compose affine + elastic deformation grid (shared for both maps)
            grid = self._make_affine_elastic_grid()

            # Apply same deformation to both label maps (nearest-neighbor)
            fine_def = F.grid_sample(fine_vol, grid, mode='nearest',
                                     padding_mode='border', align_corners=False)
            coarse_def = F.grid_sample(coarse_vol, grid, mode='nearest',
                                       padding_mode='border', align_corners=False)

            fine_long = fine_def.squeeze(0).squeeze(0).long()
            coarse_long = coarse_def.squeeze(0).squeeze(0).long()

            # 2. Left-right flip with bilateral label swap (50%)
            if torch.rand(1, device=self.device).item() < 0.5:
                fine_long = torch.flip(fine_long, [FLIP_AXIS])
                coarse_long = torch.flip(coarse_long, [FLIP_AXIS])
                # Swap bilateral labels in coarse target
                coarse_long = self.swap_lut[coarse_long]

            # 3. Intensity sampling from fine labels (background-aware)
            image = self._sample_intensities(fine_long)

            # 4. Bias field
            image = self._add_bias_field(image)

            # 5. Blur
            image = self._add_blur(image)

            # 6. Gamma augmentation (always applied)
            image = self._gamma_augment(image)

            # 7. Normalize to [0, 1]
            vmin, vmax = image.min(), image.max()
            image = (image - vmin) / (vmax - vmin + 1e-8)

            images.append(image.unsqueeze(0))
            labels.append(coarse_long)

        return torch.stack(images), torch.stack(labels)

    def _make_affine_elastic_grid(self) -> torch.Tensor:
        """Compose random affine + elastic deformation into a single sampling grid."""
        D, H, W = self.target_shape
        dev = self.device

        # --- Affine: rotation (+-15deg per axis) + scaling (0.8-1.2x) ---
        angles = (torch.rand(3, device=dev) * 2 - 1) * (15.0 * math.pi / 180.0)
        cx, cy, cz = torch.cos(angles)
        sx, sy, sz = torch.sin(angles)

        Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], device=dev, dtype=torch.float32)
        Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], device=dev, dtype=torch.float32)
        Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], device=dev, dtype=torch.float32)
        R = Rz @ Ry @ Rx

        s = torch.rand(3, device=dev) * 0.4 + 0.8  # U(0.8, 1.2)
        S = torch.diag(s)

        A = R @ S
        theta = torch.zeros(1, 3, 4, device=dev)
        theta[0, :3, :3] = A

        affine_grid = F.affine_grid(theta, [1, 1, D, H, W], align_corners=False)

        # --- Elastic deformation ---
        scale = self.cfg.get("deform_scale", 3.0)
        flow = torch.randn(1, 3, *self.flow_shape, device=dev) * scale
        flow = F.interpolate(flow, size=(D, H, W), mode='trilinear', align_corners=False)

        flow_grid = flow.permute(0, 2, 3, 4, 1)       # (1, D, H, W, 3)
        flow_grid = flow_grid[..., [2, 1, 0]]           # D,H,W -> W,H,D for grid_sample
        flow_grid = flow_grid * self.norm_factor

        # Compose: affine grid + elastic displacement
        return affine_grid + flow_grid

    def _sample_intensities(self, label_map: torch.Tensor) -> torch.Tensor:
        """Sample random intensity per fine label with background-aware handling."""
        mu_lo, mu_hi = self.cfg.get("intensity_mu_range", [0, 300])
        sig_lo, sig_hi = 0, 30  # wider range: [0, 30]

        # Standard per-label sampling
        means = torch.empty(self.n_fine_labels, device=self.device).uniform_(mu_lo, mu_hi)
        stds = torch.empty(self.n_fine_labels, device=self.device).uniform_(sig_lo, sig_hi)

        # Background-aware: override label 0 intensity
        bg_roll = torch.rand(1, device=self.device).item()
        if bg_roll < 0.3:
            # 30%: black background
            means[0] = 0.0
            stds[0] = 0.0
        elif bg_roll < 0.7:
            # 40%: dim background
            means[0] = torch.empty(1, device=self.device).uniform_(0, 20).item()
            stds[0] = torch.empty(1, device=self.device).uniform_(0, 5).item()
        # else 30%: standard random (already set)

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
        """SynthSeg gamma: gamma = exp(N(0, 0.5)), always applied."""
        z = torch.randn(1, device=self.device).item() * 0.5
        gamma = math.exp(z)
        # Shift to positive for power transform
        mn = image.min()
        image = image - mn
        mx = image.max() + 1e-8
        image = torch.pow(image / mx, gamma) * mx
        return image + mn
