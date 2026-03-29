"""
SynthSeg-style synthetic image generator for SPIM data.

Optimized for throughput:
- Elastic deformation runs at NATIVE volume size (before padding)
- Intensity sampling vectorized with label LUT
- Bias field / blur / noise on padded volume (cheap ops)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from typing import Optional


def conform_volume(vol: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Pad or crop volume to target_shape (center-aligned)."""
    D, H, W = target_shape
    out = np.zeros((D, H, W), dtype=vol.dtype)

    sd, sh, sw = vol.shape[:3]
    d0 = max(0, (sd - D) // 2)
    h0 = max(0, (sh - H) // 2)
    w0 = max(0, (sw - W) // 2)

    od = max(0, (D - sd) // 2)
    oh = max(0, (H - sh) // 2)
    ow = max(0, (W - sw) // 2)

    copy_d = min(D, sd)
    copy_h = min(H, sh)
    copy_w = min(W, sw)

    out[od:od+copy_d, oh:oh+copy_h, ow:ow+copy_w] = \
        vol[d0:d0+copy_d, h0:h0+copy_h, w0:w0+copy_w]

    return out


class SynthSegSPIMGenerator(Dataset):
    """
    On-the-fly synthetic image generator from label maps.

    Key optimization: elastic deformation runs at native volume size
    (before padding to target_shape), saving 30-50% compute since most
    volumes are smaller than the padded target.
    """

    def __init__(
        self,
        label_paths: list[str | Path],
        n_labels: int,
        target_shape: tuple[int, int, int] = (128, 128, 128),
        samples_per_epoch: int = 500,
        cfg_synth: Optional[dict] = None,
    ):
        self.label_paths = [Path(p) for p in label_paths]
        self.n_labels = n_labels
        self.target_shape = target_shape
        self.samples_per_epoch = samples_per_epoch
        self.cfg = cfg_synth or {}

        # Pre-load label maps at NATIVE size (not padded) — deform happens here
        print(f"Loading {len(self.label_paths)} label maps into memory...")
        self.label_maps = []
        for p in self.label_paths:
            vol = nib.load(str(p)).get_fdata().astype(np.int32)
            self.label_maps.append(vol)
        print(f"  Done. Shapes range from {min(v.shape[0] for v in self.label_maps)}x... "
              f"to {max(v.shape[0] for v in self.label_maps)}x...")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        rng = np.random.default_rng()

        # 1. Pick random label map (native size)
        lm_idx = rng.integers(0, len(self.label_maps))
        label_map = self.label_maps[lm_idx].copy()

        # 2. Elastic deform at NATIVE size (much smaller than target_shape)
        label_map = self._random_elastic_deform(label_map, rng)

        # 3. Pad/crop to target_shape
        label_map = conform_volume(label_map, self.target_shape)

        # 4. Generate synthetic image (vectorized)
        image = self._generate_synthetic_image(label_map, rng)

        # 5. Augmentations (cheap on padded volume — mostly element-wise)
        image = self._add_bias_field(image, rng)
        image = self._add_noise(image, rng)
        image = self._add_blur(image, rng)
        image = self._gamma_augment(image, rng)

        # 6. Normalize to [0, 1]
        vmin, vmax = image.min(), image.max()
        image = (image - vmin) / (vmax - vmin + 1e-8)

        # To tensors
        image_t = torch.from_numpy(image[np.newaxis]).float()
        label_t = torch.from_numpy(label_map).long()

        return image_t, label_t

    def _generate_synthetic_image(self, label_map: np.ndarray, rng) -> np.ndarray:
        """Vectorized intensity sampling using a LUT instead of per-label loop."""
        mu_lo, mu_hi = self.cfg.get("intensity_mu_range", [0, 300])
        sig_lo, sig_hi = self.cfg.get("intensity_sigma_range", [5, 30])

        # Build intensity LUT: random mean per label
        means = rng.uniform(mu_lo, mu_hi, size=self.n_labels).astype(np.float32)
        stds = rng.uniform(sig_lo, sig_hi, size=self.n_labels).astype(np.float32)

        # Map labels to means (vectorized lookup)
        image = means[label_map]

        # Add per-voxel noise scaled by per-label std
        noise_scale = stds[label_map]
        image += rng.standard_normal(label_map.shape).astype(np.float32) * noise_scale

        return image

    def _random_elastic_deform(self, label_map: np.ndarray, rng) -> np.ndarray:
        """Elastic deformation at native volume size (before padding)."""
        scale = self.cfg.get("deform_scale", 3.0)
        spacing = self.cfg.get("deform_grid_spacing", 32)

        shape = label_map.shape
        grid_shape = tuple(max(1, s // spacing) for s in shape)
        zoom_factors = tuple(s / g for s, g in zip(shape, grid_shape))

        # 3 separate coarse->full zooms (much faster than 4D zoom)
        dx = zoom(rng.normal(0, scale, grid_shape).astype(np.float32), zoom_factors, order=3)
        dy = zoom(rng.normal(0, scale, grid_shape).astype(np.float32), zoom_factors, order=3)
        dz = zoom(rng.normal(0, scale, grid_shape).astype(np.float32), zoom_factors, order=3)

        # Broadcasting coords (avoids huge mgrid allocation)
        d, h, w = shape
        coords = np.empty((3, d, h, w), dtype=np.float32)
        coords[0] = np.arange(d, dtype=np.float32).reshape(-1, 1, 1) + dx[:d, :h, :w]
        coords[1] = np.arange(h, dtype=np.float32).reshape(1, -1, 1) + dy[:d, :h, :w]
        coords[2] = np.arange(w, dtype=np.float32).reshape(1, 1, -1) + dz[:d, :h, :w]

        deformed = map_coordinates(label_map, coords, order=0, mode='nearest')
        return deformed.astype(np.int32)

    def _add_bias_field(self, image: np.ndarray, rng) -> np.ndarray:
        """Smooth multiplicative bias field."""
        n_coeffs = self.cfg.get("bias_field_coeffs", 4)
        strength = self.cfg.get("bias_field_strength", 0.4)

        shape = image.shape
        small = rng.normal(0, 1, (n_coeffs, n_coeffs, n_coeffs)).astype(np.float32)
        bias = zoom(small, [s / n_coeffs for s in shape], order=3)
        bias = bias[:shape[0], :shape[1], :shape[2]]
        bias = 1.0 + strength * (bias / (np.abs(bias).max() + 1e-8))

        return image * bias

    def _add_noise(self, image: np.ndarray, rng) -> np.ndarray:
        """Additive Gaussian noise."""
        lo, hi = self.cfg.get("noise_std_range", [0, 25])
        std = rng.uniform(lo, hi)
        if std > 0.1:
            image = image + rng.normal(0, std, image.shape).astype(np.float32)
        return image

    def _add_blur(self, image: np.ndarray, rng) -> np.ndarray:
        """Random Gaussian blur."""
        lo, hi = self.cfg.get("blur_sigma_range", [0, 1.5])
        sigma = rng.uniform(lo, hi)
        if sigma > 0.1:
            image = gaussian_filter(image, sigma=sigma)
        return image

    def _gamma_augment(self, image: np.ndarray, rng) -> np.ndarray:
        """Random gamma transform."""
        lo, hi = self.cfg.get("gamma_range", [0.7, 1.5])
        gamma = rng.uniform(lo, hi)
        mn = image.min()
        image = image - mn
        mx = image.max() + 1e-8
        image = np.power(image / mx, gamma) * mx
        return image + mn


class RealImageDataset(Dataset):
    """Real image + label pairs for validation. Preloaded into memory."""

    def __init__(
        self,
        image_paths: list[str | Path],
        label_paths: list[str | Path],
        target_shape: tuple[int, int, int] = (128, 128, 128),
        olfactory_labels: list[int] = None,
    ):
        assert len(image_paths) == len(label_paths)
        self.target_shape = target_shape
        olf = olfactory_labels or [1, 2]

        # Preload and conform — validation set is small (10 subjects)
        self.images = []
        self.labels = []
        for ip, lp in zip(image_paths, label_paths):
            img = nib.load(str(ip)).get_fdata().astype(np.float32)
            lab = nib.load(str(lp)).get_fdata().astype(np.int32)
            # Merge olfactory bulb to background
            for ol in olf:
                lab[lab == ol] = 0
            img = conform_volume(img, target_shape)
            lab = conform_volume(lab, target_shape)
            # Normalize
            vmin, vmax = img.min(), img.max()
            img = (img - vmin) / (vmax - vmin + 1e-8)
            self.images.append(img)
            self.labels.append(lab)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_t = torch.from_numpy(self.images[idx][np.newaxis]).float()
        label_t = torch.from_numpy(self.labels[idx]).long()
        return image_t, label_t
