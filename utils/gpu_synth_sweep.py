"""Sweepable GPU synthetic generator with all augmentations.

Extends gpu_synth_fine_v2 with:
1. Sparse label activation (fluorescence-aware: some regions dark)
2. Soft label boundaries (Gaussian blur on intensity image)
3. Spatially-varying intra-region texture (multiplicative smooth noise)

All 14 hyperparameters are controlled via the cfg dict for Optuna sweeping.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def conform_and_to_tensor(vol_np, target_shape):
    D, H, W = target_shape
    out = np.zeros((D, H, W), dtype=np.float32)
    sd, sh, sw = vol_np.shape[:3]
    d0, h0, w0 = max(0, (sd-D)//2), max(0, (sh-H)//2), max(0, (sw-W)//2)
    od, oh, ow = max(0, (D-sd)//2), max(0, (H-sh)//2), max(0, (W-sw)//2)
    cd, ch, cw = min(D, sd), min(H, sh), min(W, sw)
    out[od:od+cd, oh:oh+ch, ow:ow+cw] = vol_np[d0:d0+cd, h0:h0+ch, w0:w0+cw]
    return torch.from_numpy(out).unsqueeze(0).unsqueeze(0)


BILATERAL_PAIRS = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12),
                   (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]
FLIP_AXIS = 0


class GPUSynthSweep:
    """GPU synth generator with all augmentations, fully parameterized for sweeping."""

    def __init__(self, fine_labels_np_or_tensors, coarse_labels_np_or_tensors, n_labels,
                 target_shape, device, cfg=None, n_fine_labels=None,
                 _preloaded=False):
        self.n_labels = n_labels
        self.target_shape = target_shape
        self.device = device
        cfg = cfg or {}
        self.cfg = cfg

        if _preloaded:
            # Already GPU tensors + n_fine_labels provided — skip all loading
            self.fine_maps = fine_labels_np_or_tensors
            self.coarse_maps = coarse_labels_np_or_tensors
            self.n_fine_labels = n_fine_labels
        else:
            fine_labels_np = fine_labels_np_or_tensors
            coarse_labels_np = coarse_labels_np_or_tensors
            assert len(fine_labels_np) == len(coarse_labels_np)

            olf_labels = cfg.get("olfactory_labels", [1, 2])

            if n_fine_labels is not None:
                self.n_fine_labels = n_fine_labels
            else:
                all_fine = set()
                for f in fine_labels_np:
                    all_fine.update(np.unique(f).tolist())
                self.n_fine_labels = max(all_fine) + 1

            self.fine_maps = []
            self.coarse_maps = []
            for fnp, cnp in zip(fine_labels_np, coarse_labels_np):
                ft = conform_and_to_tensor(fnp, target_shape).to(device)
                self.fine_maps.append(ft)
                cm = cnp.copy()
                for ol in olf_labels:
                    cm[cm == ol] = 0
                ct = conform_and_to_tensor(cm, target_shape).to(device)
                self.coarse_maps.append(ct)

        D, H, W = target_shape
        self.base_grid = F.affine_grid(
            torch.eye(3, 4, device=device).unsqueeze(0),
            [1, 1, D, H, W], align_corners=False)

        spacing = cfg.get("deform_grid_spacing", 32)
        self.flow_shape = (max(1, D//spacing), max(1, H//spacing), max(1, W//spacing))
        self.norm_factor = torch.tensor(
            [2.0/W, 2.0/H, 2.0/D], device=device).reshape(1, 1, 1, 1, 3)

        swap_lut = torch.arange(n_labels, device=device, dtype=torch.long)
        for l, r in BILATERAL_PAIRS:
            if l < n_labels and r < n_labels:
                swap_lut[l] = r
                swap_lut[r] = l
        self.swap_lut = swap_lut

    @torch.no_grad()
    def generate(self, batch_size=1):
        """Full pipeline: deform -> flip -> sparse intensity -> texture -> boundary blur -> bias -> gamma -> normalize."""
        images, labels = [], []
        c = self.cfg

        for _ in range(batch_size):
            idx = torch.randint(len(self.fine_maps), (1,)).item()
            fine_vol = self.fine_maps[idx]
            coarse_vol = self.coarse_maps[idx]

            # 1. Affine + elastic deformation
            grid = self._make_affine_elastic_grid()
            fine_def = F.grid_sample(fine_vol, grid, mode='nearest',
                                     padding_mode='border', align_corners=False)
            coarse_def = F.grid_sample(coarse_vol, grid, mode='nearest',
                                       padding_mode='border', align_corners=False)
            fine_long = fine_def.squeeze(0).squeeze(0).long()
            coarse_long = coarse_def.squeeze(0).squeeze(0).long()

            # 2. LR flip (50%)
            if torch.rand(1, device=self.device).item() < 0.5:
                fine_long = torch.flip(fine_long, [FLIP_AXIS])
                coarse_long = torch.flip(coarse_long, [FLIP_AXIS])
                coarse_long = self.swap_lut[coarse_long]

            # 3. Sparse intensity sampling
            image = self._sample_intensities_sparse(fine_long, coarse_long)

            # 4. Intra-region texture
            tex_prob = c.get("texture_prob", 0.5)
            if torch.rand(1, device=self.device).item() < tex_prob:
                image = self._add_texture(image)

            # 5. Boundary blur
            bb_prob = c.get("boundary_blur_prob", 0.5)
            if torch.rand(1, device=self.device).item() < bb_prob:
                sigma_max = c.get("boundary_blur_sigma_max", 1.5)
                sigma = torch.empty(1, device=self.device).uniform_(0.5, sigma_max).item()
                image = self._gaussian_blur(image, sigma)

            # 6. Bias field
            image = self._add_bias_field(image)

            # 7. Gamma
            image = self._gamma_augment(image)

            # 8. Normalize
            vmin, vmax = image.min(), image.max()
            image = (image - vmin) / (vmax - vmin + 1e-8)

            images.append(image.unsqueeze(0))
            labels.append(coarse_long)

        return torch.stack(images), torch.stack(labels)

    def _make_affine_elastic_grid(self):
        D, H, W = self.target_shape
        dev = self.device
        angles = (torch.rand(3, device=dev) * 2 - 1) * (15.0 * math.pi / 180.0)
        cx, cy, cz = torch.cos(angles)
        sx, sy, sz = torch.sin(angles)
        Rx = torch.tensor([[1,0,0],[0,cx,-sx],[0,sx,cx]], device=dev, dtype=torch.float32)
        Ry = torch.tensor([[cy,0,sy],[0,1,0],[-sy,0,cy]], device=dev, dtype=torch.float32)
        Rz = torch.tensor([[cz,-sz,0],[sz,cz,0],[0,0,1]], device=dev, dtype=torch.float32)
        R = Rz @ Ry @ Rx
        s = torch.rand(3, device=dev) * 0.4 + 0.8
        A = R @ torch.diag(s)
        theta = torch.zeros(1, 3, 4, device=dev)
        theta[0, :3, :3] = A
        affine_grid = F.affine_grid(theta, [1, 1, D, H, W], align_corners=False)

        scale = self.cfg.get("deform_scale", 3.0)
        flow = torch.randn(1, 3, *self.flow_shape, device=dev) * scale
        flow = F.interpolate(flow, size=(D, H, W), mode='trilinear', align_corners=False)
        flow_grid = flow.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]] * self.norm_factor
        return affine_grid + flow_grid

    def _sample_intensities_sparse(self, fine_long, coarse_long):
        """Intensity sampling with sparse label activation based on coarse labels."""
        c = self.cfg
        mu_hi = c.get("intensity_mean_hi", 300)
        std_lo = c.get("intensity_std_lo", 0)
        std_hi = c.get("intensity_std_hi", 30)
        act_lo = c.get("activation_prob_lo", 0.3)
        act_hi = c.get("activation_prob_hi", 0.8)
        inact_mu_max = c.get("inactive_intensity_max", 20)
        inact_std_max = c.get("inactive_std_max", 5)
        dev = self.device

        # Determine which coarse labels are active
        p_active = torch.empty(1, device=dev).uniform_(act_lo, act_hi).item()
        coarse_active = torch.zeros(self.n_labels, device=dev, dtype=torch.bool)
        coarse_active[0] = False  # background always inactive
        for lab in range(1, self.n_labels):
            coarse_active[lab] = torch.rand(1, device=dev).item() < p_active

        # Build per-voxel activation mask from coarse labels
        voxel_active = coarse_active[coarse_long]  # (D, H, W) bool

        # Sample intensities for ALL fine labels
        means = torch.empty(self.n_fine_labels, device=dev).uniform_(0, mu_hi)
        stds = torch.empty(self.n_fine_labels, device=dev).uniform_(std_lo, std_hi)

        # Inactive overrides: low intensity for fine labels in inactive coarse regions
        inactive_means = torch.empty(self.n_fine_labels, device=dev).uniform_(0, inact_mu_max)
        inactive_stds = torch.empty(self.n_fine_labels, device=dev).uniform_(0, inact_std_max)

        # Build image
        mean_map_active = means[fine_long]
        std_map_active = stds[fine_long]
        mean_map_inactive = inactive_means[fine_long]
        std_map_inactive = inactive_stds[fine_long]

        mean_map = torch.where(voxel_active, mean_map_active, mean_map_inactive)
        std_map = torch.where(voxel_active, std_map_active, std_map_inactive)

        # Background handling (same as v2)
        bg_mask = coarse_long == 0
        bg_roll = torch.rand(1, device=dev).item()
        if bg_roll < 0.3:
            mean_map[bg_mask] = 0.0
            std_map[bg_mask] = 0.0
        elif bg_roll < 0.7:
            mean_map[bg_mask] = torch.empty(1, device=dev).uniform_(0, 20).item()
            std_map[bg_mask] = torch.empty(1, device=dev).uniform_(0, 5).item()

        return torch.normal(mean_map, std_map.clamp(min=0))

    def _add_texture(self, image):
        """Multiplicative spatially-varying texture noise."""
        c = self.cfg
        amp = c.get("texture_noise_amplitude", 0.2)
        res = c.get("texture_base_resolution", 16)
        D, H, W = self.target_shape
        small = torch.randn(1, 1, res, res, res, device=self.device)
        field = F.interpolate(small, size=(D, H, W), mode='trilinear',
                              align_corners=False).squeeze(0).squeeze(0)
        return image * (1.0 + amp * field)

    def _gaussian_blur(self, image, sigma):
        """Separable 3D Gaussian blur."""
        k_size = int(4 * sigma + 0.5) * 2 + 1
        k_size = max(k_size, 3)
        x = torch.arange(k_size, device=self.device, dtype=torch.float32) - k_size // 2
        k1d = torch.exp(-0.5 * x**2 / (sigma**2 + 1e-8))
        k1d = k1d / k1d.sum()
        img = image.unsqueeze(0).unsqueeze(0)
        pad = k_size // 2
        img = F.conv3d(img, k1d.reshape(1,1,-1,1,1), padding=(pad,0,0))
        img = F.conv3d(img, k1d.reshape(1,1,1,-1,1), padding=(0,pad,0))
        img = F.conv3d(img, k1d.reshape(1,1,1,1,-1), padding=(0,0,pad))
        return img.squeeze(0).squeeze(0)

    def _add_bias_field(self, image):
        n = self.cfg.get("bias_field_coeffs", 4)
        strength = self.cfg.get("bias_field_std", 0.4)
        D, H, W = self.target_shape
        small = torch.randn(1, 1, n, n, n, device=self.device)
        bias = F.interpolate(small, size=(D, H, W), mode='trilinear', align_corners=False)
        bias = bias.squeeze(0).squeeze(0)
        bias = 1.0 + strength * (bias / (bias.abs().max() + 1e-8))
        return image * bias

    def _gamma_augment(self, image):
        gamma_std = self.cfg.get("gamma_std", 0.5)
        z = torch.randn(1, device=self.device).item() * gamma_std
        gamma = math.exp(z)
        mn = image.min()
        image = image - mn
        mx = image.max() + 1e-8
        image = torch.pow(image / mx, gamma) * mx
        return image + mn
