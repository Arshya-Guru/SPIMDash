#!/usr/bin/env python3
"""
evaluate.py - Evaluate predicted segmentations against SPIMquant-derived labels.

Computes per-region Dice, volume correlation, and generates QC overlay images.

Usage:
    pixi run python scripts/evaluate.py \
        --pred-dir outputs/predictions/ \
        --gt-dir /nfs/khan/trainees/apooladi/brainhack/labels \
        --image-dir /nfs/khan/trainees/apooladi/brainhack/downsampled_lighsheet5 \
        --output-dir outputs/evaluation/
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import json


def dice_score(pred: np.ndarray, target: np.ndarray, label: int) -> float:
    pred_mask = (pred == label)
    true_mask = (target == label)
    if pred_mask.sum() == 0 and true_mask.sum() == 0:
        return float("nan")
    intersection = (pred_mask & true_mask).sum()
    return 2 * intersection / (pred_mask.sum() + true_mask.sum() + 1e-8)


def evaluate_subject(pred_path: Path, gt_path: Path) -> dict:
    """Compute per-label Dice and volume for one subject."""
    pred = nib.load(str(pred_path)).get_fdata().astype(np.int32)
    gt = nib.load(str(gt_path)).get_fdata().astype(np.int32)

    all_labels = sorted(set(np.unique(gt).tolist()) | set(np.unique(pred).tolist()))
    all_labels = [l for l in all_labels if l > 0]  # skip background

    results = {}
    for lab in all_labels:
        d = dice_score(pred, gt, lab)
        results[lab] = {
            "dice": d,
            "vol_pred": int((pred == lab).sum()),
            "vol_gt": int((gt == lab).sum()),
        }

    return results


def create_qc_slices(image_path: Path, pred_path: Path, gt_path: Path, out_path: Path):
    """Generate a QC PNG showing sagittal/coronal/axial slices with pred vs GT overlaid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping QC images")
        return

    img = nib.load(str(image_path)).get_fdata()
    pred = nib.load(str(pred_path)).get_fdata().astype(np.int32)
    gt = nib.load(str(gt_path)).get_fdata().astype(np.int32)

    mid = [s // 2 for s in img.shape[:3]]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    slices = [
        (img[mid[0], :, :], pred[mid[0], :, :], gt[mid[0], :, :], "Sagittal"),
        (img[:, mid[1], :], pred[:, mid[1], :], gt[:, mid[1], :], "Coronal"),
        (img[:, :, mid[2]], pred[:, :, mid[2]], gt[:, :, mid[2]], "Axial"),
    ]

    for col, (img_sl, pred_sl, gt_sl, title) in enumerate(slices):
        axes[0, col].imshow(img_sl.T, cmap="gray", origin="lower")
        axes[0, col].imshow(pred_sl.T, alpha=0.3, cmap="nipy_spectral", origin="lower")
        axes[0, col].set_title(f"Predicted - {title}")
        axes[0, col].axis("off")

        axes[1, col].imshow(img_sl.T, cmap="gray", origin="lower")
        axes[1, col].imshow(gt_sl.T, alpha=0.3, cmap="nipy_spectral", origin="lower")
        axes[1, col].set_title(f"SPIMquant GT - {title}")
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=str, required=True)
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--image-dir", type=str, default=None, help="For QC images")
    parser.add_argument("--lut", type=str, default=None, help="Label lookup TSV")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load LUT if available
    lut = None
    if args.lut and Path(args.lut).exists():
        lut = pd.read_csv(args.lut, sep="\t")

    # match pred to gt files
    pred_files = sorted(pred_dir.glob("*.nii*"))

    all_results = {}

    for pred_path in pred_files:
        sub = pred_path.name.split("_")[0]

        gt_candidates = list(gt_dir.glob(f"{sub}*dseg*"))
        if not gt_candidates:
            print(f"  No GT found for {sub}, skipping")
            continue
        gt_path = gt_candidates[0]

        print(f"Evaluating {sub}...")
        results = evaluate_subject(pred_path, gt_path)
        all_results[sub] = results

        # QC images
        if args.image_dir:
            img_candidates = list(Path(args.image_dir).glob(f"{sub}*"))
            if img_candidates:
                qc_path = out_dir / f"{sub}_qc.png"
                create_qc_slices(img_candidates[0], pred_path, gt_path, qc_path)

    # aggregate
    if all_results:
        all_labels = set()
        for sub_res in all_results.values():
            all_labels.update(sub_res.keys())

        summary = {}
        for lab in sorted(all_labels):
            dices = [
                sub_res[lab]["dice"]
                for sub_res in all_results.values()
                if lab in sub_res and not np.isnan(sub_res[lab]["dice"])
            ]
            if dices:
                summary[int(lab)] = {
                    "mean_dice": float(np.mean(dices)),
                    "std_dice": float(np.std(dices)),
                    "n_subjects": len(dices),
                }

        all_dices = [v["mean_dice"] for v in summary.values()]
        print(f"\nOverall mean Dice: {np.mean(all_dices):.4f} +/- {np.std(all_dices):.4f}")
        print(f"Evaluated {len(all_results)} subjects, {len(summary)} regions")

        with open(out_dir / "per_label_dice.json", "w") as f:
            json.dump(summary, f, indent=2)

        with open(out_dir / "per_subject_results.json", "w") as f:
            json.dump(
                {k: {str(kk): vv for kk, vv in v.items()} for k, v in all_results.items()},
                f, indent=2
            )

        print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
