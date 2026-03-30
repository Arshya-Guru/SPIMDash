#!/usr/bin/env python3
"""
Synthetic-vs-real coverage analysis using hand-crafted features + PCA.

Optimized: GPU FFT, pre-computed radial bins, subsampled GMM, batched NIfTI loading.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pixi run python scripts/coverage_analysis.py --gpu 1 --n-synth 200
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "/nfs/khan/trainees/apooladi/brainhack"
TARGET_SHAPE = (224, 320, 192)
N_GMM_SAMPLES = 30000  # subsample for GMM fitting (was using ALL voxels)

# ── Pre-computed radial bin indices (computed ONCE, reused for every image) ──

_RADIAL_BINS = None  # lazy init

def _get_radial_bins(shape, device):
    """Pre-compute radial frequency bin assignments. Cached after first call."""
    global _RADIAL_BINS
    if _RADIAL_BINS is not None:
        return _RADIAL_BINS

    D, H, W = shape
    # Normalized frequency coordinates: [-1, 1] along each axis
    fd = torch.linspace(-1, 1, D, device=device)
    fh = torch.linspace(-1, 1, H, device=device)
    fw = torch.linspace(-1, 1, W, device=device)
    # Broadcasting distance (no mgrid needed)
    dist = torch.sqrt(fd[:, None, None]**2 + fh[None, :, None]**2 + fw[None, None, :]**2)
    max_dist = dist.max().item()
    band_edges = torch.linspace(0, max_dist, 6, device=device)

    # Pre-compute bin index per voxel (0-4), 5 = out of range
    bins = torch.full_like(dist, 5, dtype=torch.long)
    for b in range(4, -1, -1):
        bins[dist < band_edges[b + 1]] = b

    _RADIAL_BINS = bins
    return bins


# ── GPU-accelerated feature extraction ──────────────────────

def compute_features_gpu(image_t: torch.Tensor, mask_t: torch.Tensor, device: torch.device) -> np.ndarray:
    """Compute 15 features on GPU. image_t and mask_t are (D, H, W) tensors on device."""
    mask_bool = mask_t > 0
    vals = image_t[mask_bool]
    n_brain = vals.numel()
    if n_brain < 100:
        return np.full(15, np.nan)

    # Normalize to [0,1]
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-8:
        return np.full(15, np.nan)
    vals_n = (vals - vmin) / (vmax - vmin)
    image_n = (image_t - vmin) / (vmax - vmin)

    feats = torch.zeros(15, device=device)

    # 0-3: Intensity statistics (on GPU)
    feats[0] = vals_n.mean()
    feats[1] = vals_n.std()
    # Skewness and kurtosis via moments
    centered = vals_n - feats[0]
    m2 = (centered ** 2).mean()
    m3 = (centered ** 3).mean()
    m4 = (centered ** 4).mean()
    std3 = m2 ** 1.5 + 1e-10
    std4 = m2 ** 2 + 1e-10
    feats[2] = m3 / std3  # skewness
    feats[3] = m4 / std4 - 3.0  # excess kurtosis

    # 4-8: Radial power spectrum (GPU FFT, pre-computed bins)
    masked_vol = image_n * mask_bool.float()
    fft = torch.fft.fftn(masked_vol)
    power = torch.fft.fftshift(torch.abs(fft) ** 2)
    bins = _get_radial_bins(image_t.shape, device)
    for b in range(5):
        band_mask = bins == b
        bp = power[band_mask]
        feats[4 + b] = torch.log(bp.mean() + 1e-10) if bp.numel() > 0 else -20.0

    # 9-10: Gradient magnitude (GPU finite differences)
    gx = image_n[1:, :, :] - image_n[:-1, :, :]
    gy = image_n[:, 1:, :] - image_n[:, :-1, :]
    gz = image_n[:, :, 1:] - image_n[:, :, :-1]
    # Pad to same size
    gx = torch.nn.functional.pad(gx, (0, 0, 0, 0, 0, 1))
    gy = torch.nn.functional.pad(gy, (0, 0, 0, 1, 0, 0))
    gz = torch.nn.functional.pad(gz, (0, 1, 0, 0, 0, 0))
    grad_mag = torch.sqrt(gx**2 + gy**2 + gz**2)
    grad_vals = grad_mag[mask_bool]
    feats[9] = grad_vals.mean()
    feats[10] = grad_vals.std()

    # 11-14: GMM (subsample to CPU — sklearn is fast on 30k points)
    if n_brain > N_GMM_SAMPLES:
        idx = torch.randperm(n_brain, device=device)[:N_GMM_SAMPLES]
        gmm_vals = vals_n[idx].cpu().numpy().reshape(-1, 1)
    else:
        gmm_vals = vals_n.cpu().numpy().reshape(-1, 1)

    try:
        gmm = GaussianMixture(n_components=2, random_state=42, max_iter=30, n_init=1)
        gmm.fit(gmm_vals)
        order = np.argsort(gmm.means_.ravel())
        feats[11] = gmm.means_.ravel()[order[0]]
        feats[12] = gmm.means_.ravel()[order[1]]
        feats[13] = gmm.weights_[order[0]]
        feats[14] = np.sqrt(gmm.covariances_.ravel()[order[0]])
    except Exception:
        feats[11:15] = torch.tensor([0.3, 0.7, 0.5, 0.1], device=device)

    return feats.cpu().numpy()


FEATURE_NAMES = [
    "intensity_mean", "intensity_std", "intensity_skew", "intensity_kurt",
    "freq_band0", "freq_band1", "freq_band2", "freq_band3", "freq_band4",
    "grad_mean", "grad_std",
    "gmm_mean1", "gmm_mean2", "gmm_weight1", "gmm_std1",
]


# ── Data loading ─────────────────────────────────────────────

def conform_volume(vol, target_shape):
    D, H, W = target_shape
    out = np.zeros((D, H, W), dtype=vol.dtype)
    sd, sh, sw = vol.shape[:3]
    d0, h0, w0 = max(0, (sd-D)//2), max(0, (sh-H)//2), max(0, (sw-W)//2)
    od, oh, ow = max(0, (D-sd)//2), max(0, (H-sh)//2), max(0, (W-sw)//2)
    cd, ch, cw = min(D, sd), min(H, sh), min(W, sw)
    out[od:od+cd, oh:oh+ch, ow:ow+cw] = vol[d0:d0+cd, h0:h0+ch, w0:w0+cw]
    return out


def load_and_featurize_real(stain, img_dir, mask_dir, device):
    """Load real images, compute features on GPU, return rows."""
    img_dir, mask_dir = Path(img_dir), Path(mask_dir)
    rows = []
    paths = sorted(img_dir.glob("*_SPIM.nii.gz"))
    for i, img_path in enumerate(paths):
        sid = img_path.name.split("_")[0]
        mask_path = mask_dir / f"{sid}_mask.nii.gz"
        if not mask_path.exists():
            continue
        img = conform_volume(nib.load(str(img_path)).get_fdata().astype(np.float32), TARGET_SHAPE)
        mask = conform_volume(nib.load(str(mask_path)).get_fdata().astype(np.float32), TARGET_SHAPE)

        img_t = torch.from_numpy(img).to(device)
        mask_t = torch.from_numpy(mask).to(device)
        feats = compute_features_gpu(img_t, mask_t, device)
        del img_t, mask_t

        row = {"source": "real", "experiment": "", "stain": stain, "subject_id": sid}
        for fi, fn in enumerate(FEATURE_NAMES):
            row[fn] = feats[fi]
        rows.append(row)

        if (i + 1) % 20 == 0:
            print(f"    {stain}: {i+1}/{len(paths)}")
    return rows


def generate_and_featurize_synth(experiment, n_samples, device, cfg_path):
    """Generate synthetic images and compute features on GPU. Returns rows."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    target_shape = tuple(cfg["volume"]["target_shape"])
    n_labels = cfg["volume"]["n_labels"]
    synth_cfg = cfg.get("synth", {})
    exclude = {"022", "032", "041"}

    label_dir = Path(cfg["data"]["label_dir"])
    label_paths = sorted(label_dir.glob("*.nii*"))
    label_paths = [p for p in label_paths if p.name.split("_")[0] not in exclude]

    if experiment == "A":
        from utils.gpu_synth import GPUSynthGenerator
        labels_np = [nib.load(str(p)).get_fdata().astype(np.int32) for p in label_paths]
        gen = GPUSynthGenerator(labels_np, n_labels, target_shape, device, synth_cfg)
        del labels_np
    elif experiment in ("C", "D"):
        from utils.gpu_synth_fine import GPUSynthGeneratorFine
        fine_dir = Path(cfg["data"]["fine_label_dir"])
        fine_paths = sorted(fine_dir.glob("*.nii*"))
        fine_paths = [p for p in fine_paths if p.name.split("_")[0] not in exclude]
        coarse_dict = {p.name.split("_")[0]: p for p in label_paths}
        fine_dict = {p.name.split("_")[0]: p for p in fine_paths}
        common = sorted(set(coarse_dict) & set(fine_dict))
        fine_np = [nib.load(str(fine_dict[s])).get_fdata().astype(np.int32) for s in common]
        coarse_np = [nib.load(str(coarse_dict[s])).get_fdata().astype(np.int32) for s in common]
        gen = GPUSynthGeneratorFine(fine_np, coarse_np, n_labels, target_shape, device, synth_cfg)
        del fine_np, coarse_np
    elif experiment in ("E", "F"):
        from utils.gpu_synth_fine_v2 import GPUSynthGeneratorFineV2
        fine_dir = Path(cfg["data"]["fine_label_dir"])
        fine_paths = sorted(fine_dir.glob("*.nii*"))
        fine_paths = [p for p in fine_paths if p.name.split("_")[0] not in exclude]
        coarse_dict = {p.name.split("_")[0]: p for p in label_paths}
        fine_dict = {p.name.split("_")[0]: p for p in fine_paths}
        common = sorted(set(coarse_dict) & set(fine_dict))
        fine_np = [nib.load(str(fine_dict[s])).get_fdata().astype(np.int32) for s in common]
        coarse_np = [nib.load(str(coarse_dict[s])).get_fdata().astype(np.int32) for s in common]
        gen = GPUSynthGeneratorFineV2(fine_np, coarse_np, n_labels, target_shape, device, synth_cfg)
        del fine_np, coarse_np
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    rows = []
    for i in range(n_samples):
        img, lab = gen.generate(batch_size=1)
        img_t = img[0, 0]  # already on device, (D, H, W)
        mask_t = (lab[0] > 0).float()  # brain mask from labels
        feats = compute_features_gpu(img_t, mask_t, device)

        row = {"source": "synthetic", "experiment": experiment, "stain": "",
               "subject_id": f"synth_{experiment}_{i:03d}"}
        for fi, fn in enumerate(FEATURE_NAMES):
            row[fn] = feats[fi]
        rows.append(row)

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_samples}")
    return rows


# ── Config ───────────────────────────────────────────────────

EXPERIMENTS = {
    "A": "configs/default.yaml",
    "C": "configs/experiment_c_roi198.yaml",
    "D": "configs/experiment_d_roiall.yaml",
    "E": "configs/experiment_e_roi198_augv2.yaml",
    "F": "configs/experiment_f_roiall_augv2.yaml",
}

REAL_STAINS = {
    "Abeta": f"{BASE}/downsampled_lighsheet5",
    "YoPro": f"{BASE}/YoPro_downsampled_lighsheet5",
    "Iba1": f"{BASE}/Iba1_downsampled_lighsheet5",
}

MASK_DIR = f"{BASE}/brain_masks"

SYNTH_COLORS = {"A": "#888888", "C": "#e41a1c", "D": "#ff7f7f",
                "E": "#984ea3", "F": "#e78ac3"}
REAL_COLORS = {"Abeta": "#377eb8", "YoPro": "#4daf4a", "Iba1": "#ff7f00"}


# ── Plotting ─────────────────────────────────────────────────

def plot_pca_scatter(df, pca, pc_x, pc_y, out_path, var_explained):
    fig, ax = plt.subplots(figsize=(12, 9))
    for stain, color in REAL_COLORS.items():
        sub = df[df["stain"] == stain]
        if len(sub) > 0:
            ax.scatter(sub[f"PC{pc_x+1}"], sub[f"PC{pc_y+1}"],
                       c=color, marker="o", s=60, alpha=0.8, label=f"Real: {stain}",
                       edgecolors="black", linewidths=0.5, zorder=3)
    for exp, color in SYNTH_COLORS.items():
        sub = df[df["experiment"] == exp]
        if len(sub) > 0:
            ax.scatter(sub[f"PC{pc_x+1}"], sub[f"PC{pc_y+1}"],
                       c=color, marker="x", s=30, alpha=0.3, label=f"Synth: Exp {exp}",
                       linewidths=1, zorder=2)
    ax.set_xlabel(f"PC{pc_x+1} ({var_explained[pc_x]:.1f}% var)", fontsize=13)
    ax.set_ylabel(f"PC{pc_y+1} ({var_explained[pc_y]:.1f}% var)", fontsize=13)
    ax.set_title(f"Synthetic vs Real Coverage — PC{pc_x+1} vs PC{pc_y+1}", fontsize=14)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_boxplots(df, feature_names, out_path):
    sources = []
    for stain in REAL_COLORS:
        if (df["stain"] == stain).any():
            sources.append(("stain", stain))
    for exp in SYNTH_COLORS:
        if (df["experiment"] == exp).any():
            sources.append(("experiment", exp))

    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    axes = axes.flatten()
    for fi, fname in enumerate(feature_names):
        ax = axes[fi]
        data, labels, colors = [], [], []
        for col, val in sources:
            sub = df[df[col] == val][fname].dropna()
            if len(sub) > 0:
                data.append(sub.values)
                labels.append(val if col == "stain" else f"Exp {val}")
                colors.append(REAL_COLORS.get(val, SYNTH_COLORS.get(val, "gray")))
        if data:
            bp = ax.boxplot(data, patch_artist=True, widths=0.6)
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(fname, fontsize=9)
        ax.grid(True, alpha=0.2)
    plt.suptitle("Per-Feature Comparison: Real vs Synthetic", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_coverage_cdf(df, out_path):
    from scipy.spatial.distance import cdist
    real_mask = df["source"] == "real"
    real_feats = df.loc[real_mask, FEATURE_NAMES].values
    if len(real_feats) < 2:
        return None

    real_dists = cdist(real_feats, real_feats)
    np.fill_diagonal(real_dists, np.inf)
    real_nn = real_dists.min(axis=1)
    ref_median = np.median(real_nn)

    fig, ax = plt.subplots(figsize=(10, 7))
    sorted_rr = np.sort(real_nn)
    ax.plot(sorted_rr, np.linspace(0, 1, len(sorted_rr)),
            color="black", linewidth=2, linestyle="--", label="Real-Real NN", zorder=5)
    ax.axvline(ref_median, color="black", linestyle=":", alpha=0.5,
               label=f"Real-Real median={ref_median:.2f}")

    coverage_summary = {}
    for exp, color in SYNTH_COLORS.items():
        synth_mask = df["experiment"] == exp
        synth_feats = df.loc[synth_mask, FEATURE_NAMES].values
        if len(synth_feats) == 0:
            continue
        dists = cdist(real_feats, synth_feats)
        nn_dists = dists.min(axis=1)
        sorted_d = np.sort(nn_dists)
        ax.plot(sorted_d, np.linspace(0, 1, len(sorted_d)),
                color=color, linewidth=2, label=f"Exp {exp}", alpha=0.8)
        coverage_summary[exp] = (nn_dists <= ref_median).mean()

    ax.set_xlabel("Distance to nearest synthetic neighbor (standardized)", fontsize=13)
    ax.set_ylabel("Fraction of real images", fontsize=13)
    ax.set_title("Coverage CDF: How well do synthetic images cover real data?", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return coverage_summary


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-synth", type=int, default=200)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path("outputs/coverage_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    t_total = time.time()

    # ── 1. Real images ──
    print("\n=== Real images ===")
    for stain, img_dir in REAL_STAINS.items():
        if not Path(img_dir).exists():
            print(f"  {stain}: not found, skipping")
            continue
        t0 = time.time()
        rows = load_and_featurize_real(stain, img_dir, MASK_DIR, device)
        all_rows.extend(rows)
        print(f"  {stain}: {len(rows)} images, {time.time()-t0:.1f}s")

    # ── 2. Synthetic images ──
    print("\n=== Synthetic images ===")
    for exp, cfg_path in EXPERIMENTS.items():
        if not Path(cfg_path).exists():
            print(f"  Exp {exp}: config not found, skipping")
            continue
        try:
            t0 = time.time()
            print(f"  Exp {exp}: generating + featurizing {args.n_synth} samples...")
            rows = generate_and_featurize_synth(exp, args.n_synth, device, cfg_path)
            all_rows.extend(rows)
            print(f"  Exp {exp}: done, {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  Exp {exp}: FAILED — {e}")
            import traceback; traceback.print_exc()

    # ── 3. DataFrame ──
    df = pd.DataFrame(all_rows)
    n_real = (df["source"] == "real").sum()
    n_synth = (df["source"] == "synthetic").sum()
    print(f"\nTotal: {len(df)} samples ({n_real} real, {n_synth} synthetic)")

    df = df.dropna(subset=FEATURE_NAMES)

    # ── 4. Standardize on real stats ──
    real_mask = df["source"] == "real"
    feat_mean = df.loc[real_mask, FEATURE_NAMES].mean()
    feat_std = df.loc[real_mask, FEATURE_NAMES].std().replace(0, 1)
    for fn in FEATURE_NAMES:
        df[fn] = (df[fn] - feat_mean[fn]) / feat_std[fn]

    # ── 5. PCA ──
    X = df[FEATURE_NAMES].values
    pca = PCA(n_components=min(5, X.shape[1]))
    pcs = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_ * 100
    for i in range(pcs.shape[1]):
        df[f"PC{i+1}"] = pcs[:, i]
    print(f"PCA variance: {[f'{v:.1f}%' for v in var_explained[:5]]}")

    # ── 6. Plots ──
    print("\n=== Plots ===")
    plot_pca_scatter(df, pca, 0, 1, out_dir / "pca_pc1_pc2.png", var_explained)
    print("  pca_pc1_pc2.png")
    plot_pca_scatter(df, pca, 0, 2, out_dir / "pca_pc1_pc3.png", var_explained)
    print("  pca_pc1_pc3.png")
    plot_feature_boxplots(df, FEATURE_NAMES, out_dir / "feature_boxplots.png")
    print("  feature_boxplots.png")
    coverage = plot_coverage_cdf(df, out_dir / "coverage_cdf.png")
    print("  coverage_cdf.png")

    # ── 7. Save ──
    df.to_csv(out_dir / "features.csv", index=False)
    loadings = pd.DataFrame(pca.components_.T, index=FEATURE_NAMES,
                            columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    loadings.to_csv(out_dir / "pca_loadings.csv")

    summary = [f"Coverage Analysis Summary", "=" * 40, "",
               f"Real images: {n_real}", f"Synthetic images: {n_synth}",
               f"PCA variance: {[f'{v:.1f}%' for v in var_explained[:5]]}", ""]
    if coverage:
        summary.append("Coverage (fraction of real within real-real median NN distance):")
        for exp, frac in sorted(coverage.items()):
            summary.append(f"  Experiment {exp}: {frac:.1%}")
    summary_text = "\n".join(summary)
    (out_dir / "summary.txt").write_text(summary_text)
    print(f"\n{summary_text}")
    print(f"\nTotal time: {time.time()-t_total:.0f}s")
    print(f"Outputs: {out_dir}/")


if __name__ == "__main__":
    main()
