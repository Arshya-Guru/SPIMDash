"""Reusable GPU feature extraction for coverage analysis and sweep.

15 features per image (computed within brain mask):
  0-3:   intensity mean, std, skewness, kurtosis
  4-8:   log power in 5 radial frequency bands
  9-10:  gradient magnitude mean, std
  11-14: percentile-based (P5, P95, P95/P50, P50/P5)

Supports caching real image features to disk.
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

BASE = "/nfs/khan/trainees/apooladi/brainhack"
TARGET_SHAPE = (224, 320, 192)

FEATURE_NAMES = [
    "intensity_mean", "intensity_std", "intensity_skew", "intensity_kurt",
    "freq_band0", "freq_band1", "freq_band2", "freq_band3", "freq_band4",
    "grad_mean", "grad_std",
    "intensity_p05", "intensity_p95", "pctile_ratio_high", "pctile_ratio_low",
]

_RADIAL_BINS = None


def _get_radial_bins(shape, device):
    global _RADIAL_BINS
    if _RADIAL_BINS is not None and _RADIAL_BINS.shape == shape:
        return _RADIAL_BINS
    D, H, W = shape
    fd = torch.linspace(-1, 1, D, device=device)
    fh = torch.linspace(-1, 1, H, device=device)
    fw = torch.linspace(-1, 1, W, device=device)
    dist = torch.sqrt(fd[:, None, None]**2 + fh[None, :, None]**2 + fw[None, None, :]**2)
    max_dist = dist.max().item()
    band_edges = torch.linspace(0, max_dist, 6, device=device)
    bins = torch.full_like(dist, 5, dtype=torch.long)
    for b in range(4, -1, -1):
        bins[dist < band_edges[b + 1]] = b
    _RADIAL_BINS = bins
    return bins


def conform_volume(vol, target_shape=TARGET_SHAPE):
    D, H, W = target_shape
    out = np.zeros((D, H, W), dtype=vol.dtype)
    sd, sh, sw = vol.shape[:3]
    d0, h0, w0 = max(0, (sd-D)//2), max(0, (sh-H)//2), max(0, (sw-W)//2)
    od, oh, ow = max(0, (D-sd)//2), max(0, (H-sh)//2), max(0, (W-sw)//2)
    cd, ch, cw = min(D, sd), min(H, sh), min(W, sw)
    out[od:od+cd, oh:oh+ch, ow:ow+cw] = vol[d0:d0+cd, h0:h0+ch, w0:w0+cw]
    return out


@torch.no_grad()
def compute_features_gpu(image_t: torch.Tensor, mask_t: torch.Tensor,
                         device: torch.device) -> np.ndarray:
    """Compute 15 features on GPU. image_t and mask_t are (D, H, W) on device."""
    mask_bool = mask_t > 0
    vals = image_t[mask_bool]
    n_brain = vals.numel()
    if n_brain < 100:
        return np.full(15, np.nan)

    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-8:
        return np.full(15, np.nan)
    vals_n = (vals - vmin) / (vmax - vmin)
    image_n = (image_t - vmin) / (vmax - vmin)

    feats = torch.zeros(15, device=device)

    # 0-3: Intensity statistics
    feats[0] = vals_n.mean()
    feats[1] = vals_n.std()
    centered = vals_n - feats[0]
    m2 = (centered ** 2).mean()
    m3 = (centered ** 3).mean()
    m4 = (centered ** 4).mean()
    feats[2] = m3 / (m2 ** 1.5 + 1e-10)
    feats[3] = m4 / (m2 ** 2 + 1e-10) - 3.0

    # 4-8: Radial power spectrum
    masked_vol = image_n * mask_bool.float()
    fft = torch.fft.fftn(masked_vol)
    power = torch.fft.fftshift(torch.abs(fft) ** 2)
    bins = _get_radial_bins(image_t.shape, device)
    for b in range(5):
        bp = power[bins == b]
        feats[4 + b] = torch.log(bp.mean() + 1e-10) if bp.numel() > 0 else -20.0

    # 9-10: Gradient magnitude
    gx = F.pad(image_n[1:, :, :] - image_n[:-1, :, :], (0, 0, 0, 0, 0, 1))
    gy = F.pad(image_n[:, 1:, :] - image_n[:, :-1, :], (0, 0, 0, 1, 0, 0))
    gz = F.pad(image_n[:, :, 1:] - image_n[:, :, :-1], (0, 1, 0, 0, 0, 0))
    grad_vals = torch.sqrt(gx**2 + gy**2 + gz**2)[mask_bool]
    feats[9] = grad_vals.mean()
    feats[10] = grad_vals.std()

    # 11-14: Percentile-based features (replace broken GMM)
    sorted_vals = torch.sort(vals_n).values
    n = sorted_vals.numel()
    p05 = sorted_vals[int(n * 0.05)]
    p50 = sorted_vals[int(n * 0.50)]
    p95 = sorted_vals[int(n * 0.95)]
    feats[11] = p05
    feats[12] = p95
    feats[13] = p95 / (p50 + 1e-8)  # high ratio
    feats[14] = p50 / (p05 + 1e-8)  # low ratio (capped later if needed)

    return feats.cpu().numpy()


def load_real_features(device, cache_path="outputs/sweep/real_features.npy"):
    """Load or compute features for all real images. Returns (features, stain_labels)."""
    cache_path = Path(cache_path)
    stain_cache = cache_path.with_suffix(".stains.npy")

    if cache_path.exists() and stain_cache.exists():
        print(f"  Loading cached real features from {cache_path}")
        return np.load(cache_path), np.load(stain_cache, allow_pickle=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    stains = {
        "Abeta": f"{BASE}/downsampled_lighsheet5",
        "YoPro": f"{BASE}/YoPro_downsampled_lighsheet5",
        "Iba1": f"{BASE}/Iba1_downsampled_lighsheet5",
    }
    mask_dir = Path(f"{BASE}/brain_masks")

    all_feats = []
    all_stains = []
    for stain, img_dir in stains.items():
        img_dir = Path(img_dir)
        if not img_dir.exists():
            continue
        paths = sorted(img_dir.glob("*_SPIM.nii.gz"))
        for ip in paths:
            sid = ip.name.split("_")[0]
            mp = mask_dir / f"{sid}_mask.nii.gz"
            if not mp.exists():
                continue
            img = conform_volume(nib.load(str(ip)).get_fdata().astype(np.float32))
            mask = conform_volume(nib.load(str(mp)).get_fdata().astype(np.float32))
            img_t = torch.from_numpy(img).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            feats = compute_features_gpu(img_t, mask_t, device)
            del img_t, mask_t
            all_feats.append(feats)
            all_stains.append(stain)
        print(f"  {stain}: {sum(1 for s in all_stains if s == stain)} images")

    feats_arr = np.stack(all_feats)
    stains_arr = np.array(all_stains)
    np.save(cache_path, feats_arr)
    np.save(stain_cache, stains_arr)
    print(f"  Saved {len(feats_arr)} real features to {cache_path}")
    return feats_arr, stains_arr


def compute_real_stats(real_feats, cache_path="outputs/sweep/real_stats.npy"):
    """Compute mean/std per feature from real images. Returns (mean, std)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        stats = np.load(cache_path)
        return stats[0], stats[1]
    mean = np.nanmean(real_feats, axis=0)
    std = np.nanstd(real_feats, axis=0)
    std[std < 1e-8] = 1.0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, np.stack([mean, std]))
    return mean, std


def compute_real_nn_stats(real_feats, mean, std, cache_path="outputs/sweep/real_nn_stats.npy"):
    """Compute real-real NN distance stats. Returns (median, p95)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        stats = np.load(cache_path)
        return stats[0], stats[1]
    from scipy.spatial.distance import cdist
    real_std = (real_feats - mean) / std
    dists = cdist(real_std, real_std)
    np.fill_diagonal(dists, np.inf)
    nn = dists.min(axis=1)
    result = np.array([np.median(nn), np.percentile(nn, 95)])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, result)
    return result[0], result[1]
