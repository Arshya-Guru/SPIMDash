#!/usr/bin/env python3
"""Post-sweep analysis: top configs, PCA, CDFs, importance plots.

Usage:
    pixi run python scripts/sweep_analysis.py --study-name sweep_roi198 --gpu 0
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.gpu_synth_sweep import GPUSynthSweep
from utils.sweep_features import (
    compute_features_gpu, load_real_features, compute_real_stats,
    compute_real_nn_stats, FEATURE_NAMES,
)
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

BASE = "/nfs/khan/trainees/apooladi/brainhack"
TARGET_SHAPE = (224, 320, 192)
N_LABELS = 23


def load_label_maps(fine_label_dir, coarse_label_dir):
    exclude = {"022", "032", "041"}
    coarse_dict = {p.name.split("_")[0]: p for p in sorted(Path(coarse_label_dir).glob("*.nii*"))}
    fine_dict = {p.name.split("_")[0]: p for p in sorted(Path(fine_label_dir).glob("*.nii*"))}
    common = sorted(set(coarse_dict) & set(fine_dict) - exclude)
    fine_np = [nib.load(str(fine_dict[s])).get_fdata().astype(np.int32) for s in common]
    coarse_np = [nib.load(str(coarse_dict[s])).get_fdata().astype(np.int32) for s in common]
    return fine_np, coarse_np


def generate_features(cfg, fine_np, coarse_np, device, n=200):
    """Generate synthetic images with given config and compute features."""
    gen = GPUSynthSweep(fine_np, coarse_np, N_LABELS, TARGET_SHAPE, device, cfg)
    feats = []
    for i in range(n):
        imgs, labs = gen.generate(batch_size=1)
        mask = (labs[0] > 0).float()
        f = compute_features_gpu(imgs[0, 0], mask, device)
        feats.append(f)
    del gen
    torch.cuda.empty_cache()
    return np.stack(feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fine-labels", choices=["roi198", "roiall"], default="roi198")
    parser.add_argument("--n-synth", type=int, default=200)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    db_path = Path(f"outputs/sweep/{args.study_name}.db")
    out_dir = Path("outputs/sweep/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading study: {args.study_name}")
    study = optuna.load_study(study_name=args.study_name,
                              storage=f"sqlite:///{db_path}")

    # ── Top trials ──
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -999,
                    reverse=True)
    valid = [t for t in trials if t.value is not None and t.value > 0]
    print(f"\nTotal trials: {len(trials)}, valid: {len(valid)}")

    print(f"\n{'='*80}")
    print(f"Top 10 trials:")
    print(f"{'='*80}")
    for t in valid[:10]:
        print(f"  Trial {t.number:4d} | obj={t.value:.4f} | "
              f"auc={t.user_attrs.get('coverage_auc', 0):.4f} | "
              f"spread={t.user_attrs.get('spread_ratio', 0):.4f} | "
              f"disc={t.user_attrs.get('discriminability', 0):.4f} | "
              f"covered={t.user_attrs.get('pct_covered_at_median', 0):.1%}")

    # Save top 10 params
    with open(out_dir / "top10_params.txt", "w") as f:
        for t in valid[:10]:
            f.write(f"Trial {t.number} (obj={t.value:.4f}):\n")
            for k, v in t.params.items():
                f.write(f"  {k}: {v}\n")
            for k, v in t.user_attrs.items():
                f.write(f"  [{k}]: {v}\n")
            f.write("\n")

    if len(valid) < 3:
        print("Not enough valid trials for full analysis")
        return

    # ── Load real features ──
    print("\nLoading real features...")
    real_feats, real_stains = load_real_features(device)
    real_mean, real_std = compute_real_stats(real_feats)
    real_median_nn, real_p95_nn = compute_real_nn_stats(real_feats, real_mean, real_std)
    real_feats_std = (real_feats - real_mean) / real_std

    # ── Generate features for top 3 configs ──
    fine_dir = (f"{BASE}/roi_labels_198" if args.fine_labels == "roi198"
                else f"{BASE}/roi_labels_all")
    fine_np, coarse_np = load_label_maps(fine_dir, f"{BASE}/labels")

    top3_feats = {}
    for i, t in enumerate(valid[:3]):
        print(f"\nGenerating {args.n_synth} images for trial {t.number}...")
        cfg = dict(t.params)
        cfg.update({"deform_scale": 3.0, "deform_grid_spacing": 32, "bias_field_coeffs": 4})
        feats = generate_features(cfg, fine_np, coarse_np, device, args.n_synth)
        feats_valid = feats[~np.any(np.isnan(feats), axis=1)]
        top3_feats[f"Sweep #{i+1} (T{t.number})"] = (feats_valid - real_mean) / real_std
        print(f"  {len(feats_valid)} valid features")

    del fine_np, coarse_np

    # ── PCA ──
    print("\nRunning PCA...")
    all_feats = [real_feats_std]
    all_labels = ["real"] * len(real_feats_std)
    for name, sf in top3_feats.items():
        all_feats.append(sf)
        all_labels.extend([name] * len(sf))
    X = np.vstack(all_feats)

    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    # PCA scatter
    fig, ax = plt.subplots(figsize=(12, 9))
    n_real = len(real_feats_std)
    stain_colors = {"Abeta": "#377eb8", "YoPro": "#4daf4a", "Iba1": "#ff7f00"}
    for stain, color in stain_colors.items():
        mask = real_stains == stain
        idx = np.where(mask)[0]
        if len(idx) > 0:
            ax.scatter(pcs[idx, 0], pcs[idx, 1], c=color, marker="o", s=60,
                       alpha=0.8, label=f"Real: {stain}", edgecolors="black",
                       linewidths=0.5, zorder=3)

    sweep_colors = ["#e41a1c", "#984ea3", "#ff7f00"]
    offset = n_real
    for i, (name, sf) in enumerate(top3_feats.items()):
        end = offset + len(sf)
        ax.scatter(pcs[offset:end, 0], pcs[offset:end, 1], c=sweep_colors[i],
                   marker="x", s=30, alpha=0.3, label=name, linewidths=1, zorder=2)
        offset = end

    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=13)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=13)
    ax.set_title("Sweep Top 3 vs Real — PCA", fontsize=14)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "pca_top3.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved pca_top3.png")

    # ── Coverage CDF ──
    fig, ax = plt.subplots(figsize=(10, 7))
    # Real-real reference
    real_dists = cdist(real_feats_std, real_feats_std)
    np.fill_diagonal(real_dists, np.inf)
    real_nn = real_dists.min(axis=1)
    ax.plot(np.sort(real_nn), np.linspace(0, 1, len(real_nn)),
            "k--", linewidth=2, label="Real-Real NN")
    ax.axvline(np.median(real_nn), color="black", linestyle=":", alpha=0.5,
               label=f"Real median={np.median(real_nn):.2f}")

    for i, (name, sf) in enumerate(top3_feats.items()):
        dists = cdist(real_feats_std, sf)
        nn = dists.min(axis=1)
        ax.plot(np.sort(nn), np.linspace(0, 1, len(nn)),
                color=sweep_colors[i], linewidth=2, label=name)

    ax.set_xlabel("Distance to nearest synthetic neighbor", fontsize=13)
    ax.set_ylabel("Fraction of real images", fontsize=13)
    ax.set_title("Coverage CDF: Sweep Top 3 vs Real", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "coverage_cdf_top3.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved coverage_cdf_top3.png")

    # ── Optuna importance ──
    try:
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        fig.figure.set_size_inches(12, 8)
        plt.tight_layout()
        plt.savefig(out_dir / "param_importances.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved param_importances.png")
    except Exception as e:
        print(f"  Param importance plot failed: {e}")

    # ── Parallel coordinates (top 50) ──
    try:
        fig = optuna.visualization.matplotlib.plot_parallel_coordinate(
            study, params=list(valid[0].params.keys())[:8])  # first 8 params for readability
        fig.figure.set_size_inches(16, 8)
        plt.tight_layout()
        plt.savefig(out_dir / "parallel_coordinates.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved parallel_coordinates.png")
    except Exception as e:
        print(f"  Parallel coordinates plot failed: {e}")

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
