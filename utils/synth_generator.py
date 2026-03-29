"""
SynthSeg-style synthetic image generator for SPIM data.

Takes a label map and generates a synthetic image by:
1. Sampling random intensity per anatomical region
2. Applying spatial deformation
3. Adding bias field, noise, blur
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path
from scipy.ndimage import gaussian_filter, map_coordinates
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

    Each __getitem__ call:
      1. Picks a random label map from the pool
      2. Generates a synthetic image from it
      3. Returns (synthetic_image, label_map) pair

    This means we have effectively infinite training data.
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

        # defaults (overridden by config)
        self.cfg = cfg_synth or {}

        # pre-load all label maps into memory (they're small at level 5)
        print(f"Loading {len(self.label_paths)} label maps into memory...")
        self.label_maps = []
        for p in self.label_paths:
            vol = nib.load(str(p)).get_fdata().astype(np.int32)
            vol = conform_volume(vol, target_shape)
            self.label_maps.append(vol)
        print(f"  Done. Unique labels across dataset: {self._count_unique_labels()}")

    def _count_unique_labels(self) -> int:
        all_labels = set()
        for lm in self.label_maps:
            all_labels.update(np.unique(lm).tolist())
        return len(all_labels)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # pick random label map
        rng = np.random.default_rng()
        lm_idx = rng.integers(0, len(self.label_maps))
        label_map = self.label_maps[lm_idx].copy()

        # --- spatial deformation (applied to label map) ---
        label_map = self._random_elastic_deform(label_map, rng)

        # --- generate synthetic image from label map ---
        image = self._generate_synthetic_image(label_map, rng)

        # --- intensity augmentations on the synthetic image ---
        image = self._add_bias_field(image, rng)
        image = self._add_noise(image, rng)
        image = self._add_blur(image, rng)
        image = self._gamma_augment(image, rng)

        # normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # to tensors: image is (1, D, H, W), label is (D, H, W)
        image_t = torch.from_numpy(image[np.newaxis]).float()
        label_t = torch.from_numpy(label_map).long()

        return image_t, label_t

    def _generate_synthetic_image(self, label_map: np.ndarray, rng) -> np.ndarray:
        """Core SynthSeg trick: assign random intensity to each label."""
        image = np.zeros_like(label_map, dtype=np.float32)
        mu_lo, mu_hi = self.cfg.get("intensity_mu_range", [0, 300])
        sig_lo, sig_hi = self.cfg.get("intensity_sigma_range", [5, 30])

        unique_labels = np.unique(label_map)
        for lab in unique_labels:
            mu = rng.uniform(mu_lo, mu_hi)
            sigma = rng.uniform(sig_lo, sig_hi)
            mask = label_map == lab
            image[mask] = rng.normal(mu, sigma, size=mask.sum()).astype(np.float32)

        return image

    def _random_elastic_deform(self, label_map: np.ndarray, rng) -> np.ndarray:
        """
        Random elastic deformation applied to the label map.
        Uses nearest-neighbor interpolation to keep labels discrete.
        """
        scale = self.cfg.get("deform_scale", 3.0)
        spacing = self.cfg.get("deform_grid_spacing", 32)

        shape = label_map.shape
        # coarse displacement field
        grid_shape = tuple(max(1, s // spacing) for s in shape)
        dx = rng.normal(0, scale, grid_shape).astype(np.float32)
        dy = rng.normal(0, scale, grid_shape).astype(np.float32)
        dz = rng.normal(0, scale, grid_shape).astype(np.float32)

        # upsample to full resolution with smooth interpolation
        from scipy.ndimage import zoom
        zoom_factors = tuple(s / g for s, g in zip(shape, grid_shape))
        dx = zoom(dx, zoom_factors, order=3)
        dy = zoom(dy, zoom_factors, order=3)
        dz = zoom(dz, zoom_factors, order=3)

        # create sampling coordinates
        coords = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)
        coords[0] += dx
        coords[1] += dy
        coords[2] += dz

        # nearest-neighbor for label maps
        deformed = map_coordinates(label_map, coords, order=0, mode='nearest')
        return deformed.astype(np.int32)

    def _add_bias_field(self, image: np.ndarray, rng) -> np.ndarray:
        """Smooth multiplicative bias field (simulates illumination variation)."""
        n_coeffs = self.cfg.get("bias_field_coeffs", 4)
        strength = self.cfg.get("bias_field_strength", 0.4)

        shape = image.shape
        small = rng.normal(0, 1, (n_coeffs, n_coeffs, n_coeffs)).astype(np.float32)
        from scipy.ndimage import zoom
        bias = zoom(small, [s / n_coeffs for s in shape], order=3)
        bias = bias[:shape[0], :shape[1], :shape[2]]  # trim if needed
        bias = 1.0 + strength * (bias / (np.abs(bias).max() + 1e-8))

        return image * bias

    def _add_noise(self, image: np.ndarray, rng) -> np.ndarray:
        """Additive Gaussian noise."""
        lo, hi = self.cfg.get("noise_std_range", [0, 25])
        std = rng.uniform(lo, hi)
        noise = rng.normal(0, std, image.shape).astype(np.float32)
        return image + noise

    def _add_blur(self, image: np.ndarray, rng) -> np.ndarray:
        """Random Gaussian blur (simulates resolution variation)."""
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
        image = np.power(image / (image.max() + 1e-8), gamma) * (image.max() + 1e-8)
        return image + mn


class RealImageDataset(Dataset):
    """
    Simple dataset of real image + label pairs for validation.
    No synthetic generation - just loads and conforms real data.
    """

    def __init__(
        self,
        image_paths: list[str | Path],
        label_paths: list[str | Path],
        target_shape: tuple[int, int, int] = (128, 128, 128),
    ):
        assert len(image_paths) == len(label_paths)
        self.image_paths = [Path(p) for p in image_paths]
        self.label_paths = [Path(p) for p in label_paths]
        self.target_shape = target_shape

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = nib.load(str(self.image_paths[idx])).get_fdata().astype(np.float32)
        label = nib.load(str(self.label_paths[idx])).get_fdata().astype(np.int32)

        # conform to target shape
        image = conform_volume(image, self.target_shape)
        label = conform_volume(label, self.target_shape)

        # normalize image
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        image_t = torch.from_numpy(image[np.newaxis]).float()
        label_t = torch.from_numpy(label).long()

        return image_t, label_t
