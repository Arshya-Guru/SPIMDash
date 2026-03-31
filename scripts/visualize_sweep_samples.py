#!/usr/bin/env python3
"""Generate visual samples from top sweep configs + real data + Experiment A baseline.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pixi run python scripts/visualize_sweep_samples.py --gpu 0
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "/nfs/khan/trainees/apooladi/brainhack"
TARGET_SHAPE = (224, 320, 192)
N_LABELS = 23
EXCLUDE = {"022", "032", "041"}


def conform_volume(vol, target_shape=TARGET_SHAPE):
    D, H, W = target_shape
    out = np.zeros((D, H, W), dtype=vol.dtype)
    sd, sh, sw = vol.shape[:3]
    d0, h0, w0 = max(0, (sd-D)//2), max(0, (sh-H)//2), max(0, (sw-W)//2)
    od, oh, ow = max(0, (D-sd)//2), max(0, (H-sh)//2), max(0, (W-sw)//2)
    cd, ch, cw = min(D, sd), min(H, sh), min(W, sw)
    out[od:od+cd, oh:oh+ch, ow:ow+cw] = vol[d0:d0+cd, h0:h0+ch, w0:w0+cw]
    return out


def save_grid(image_np, label_np, out_path, title=""):
    """Save 2x3 grid: top=image slices (gray), bottom=label slices (color)."""
    D, H, W = image_np.shape
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Image slices
    slices_img = [
        (image_np[mid_d, :, :].T, f"Axial (d={mid_d})"),
        (image_np[:, mid_h, :].T, f"Coronal (h={mid_h})"),
        (image_np[:, :, mid_w].T, f"Sagittal (w={mid_w})"),
    ]
    for col, (sl, name) in enumerate(slices_img):
        vmin, vmax = np.percentile(sl[sl > 0], [1, 99]) if (sl > 0).any() else (0, 1)
        axes[0, col].imshow(sl, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(name, fontsize=10)
        axes[0, col].axis("off")

    # Label slices
    slices_lab = [
        label_np[mid_d, :, :].T,
        label_np[:, mid_h, :].T,
        label_np[:, :, mid_w].T,
    ]
    lab_max = max(label_np.max(), 1)
    for col, sl in enumerate(slices_lab):
        axes[1, col].imshow(sl, cmap="tab20", origin="lower", vmin=0, vmax=lab_max,
                            interpolation="nearest")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Image", fontsize=12)
    axes[1, 0].set_ylabel("Labels", fontsize=12)
    if title:
        fig.suptitle(title, fontsize=13, y=0.98)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def load_label_maps(fine_dir, coarse_dir):
    """Load fine + coarse label maps, return numpy lists."""
    coarse_paths = sorted(Path(coarse_dir).glob("*.nii*"))
    fine_paths = sorted(Path(fine_dir).glob("*.nii*"))
    coarse_dict = {p.name.split("_")[0]: p for p in coarse_paths}
    fine_dict = {p.name.split("_")[0]: p for p in fine_paths}
    common = sorted(set(coarse_dict) & set(fine_dict) - EXCLUDE)
    fine_np = [nib.load(str(fine_dict[s])).get_fdata().astype(np.int32) for s in common]
    coarse_np = [nib.load(str(coarse_dict[s])).get_fdata().astype(np.int32) for s in common]
    return fine_np, coarse_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_base = Path("outputs/sweep/samples")

    # ── Trial configs ──
    TRIALS = {
        257: {
            "activation_prob_lo": 0.156, "activation_prob_hi": 0.677,
            "inactive_intensity_max": 29.5, "inactive_std_max": 5.9,
            "boundary_blur_sigma_max": 0.553, "boundary_blur_prob": 0.447,
            "texture_noise_amplitude": 0.256, "texture_base_resolution": 16,
            "texture_prob": 0.939,
            "intensity_mean_hi": 138.6, "intensity_std_lo": 3.65, "intensity_std_hi": 22.1,
            "gamma_std": 0.958, "bias_field_std": 1.153,
            "deform_scale": 3.0, "deform_grid_spacing": 32, "bias_field_coeffs": 4,
        },
        422: {
            "activation_prob_lo": 0.201, "activation_prob_hi": 0.656,
            "inactive_intensity_max": 31.7, "inactive_std_max": 6.57,
            "boundary_blur_sigma_max": 0.502, "boundary_blur_prob": 0.440,
            "texture_noise_amplitude": 0.475, "texture_base_resolution": 15,
            "texture_prob": 0.977,
            "intensity_mean_hi": 145.6, "intensity_std_lo": 5.33, "intensity_std_hi": 22.0,
            "gamma_std": 1.224, "bias_field_std": 1.272,
            "deform_scale": 3.0, "deform_grid_spacing": 32, "bias_field_coeffs": 4,
        },
        439: {
            "activation_prob_lo": 0.190, "activation_prob_hi": 0.628,
            "inactive_intensity_max": 36.7, "inactive_std_max": 6.69,
            "boundary_blur_sigma_max": 0.591, "boundary_blur_prob": 0.431,
            "texture_noise_amplitude": 0.476, "texture_base_resolution": 16,
            "texture_prob": 0.619,
            "intensity_mean_hi": 167.8, "intensity_std_lo": 2.97, "intensity_std_hi": 20.9,
            "gamma_std": 1.251, "bias_field_std": 1.322,
            "deform_scale": 3.0, "deform_grid_spacing": 32, "bias_field_coeffs": 4,
        },
    }

    # ── 1. Load label maps for sweep generator ──
    print("Loading roi198 fine + roi22 coarse label maps...")
    fine_np, coarse_np = load_label_maps(
        f"{BASE}/roi_labels_198", f"{BASE}/labels")
    print(f"  {len(fine_np)} subjects loaded")

    # ── 2. Generate sweep trial samples ──
    from utils.gpu_synth_sweep import GPUSynthSweep

    for trial_num, cfg in TRIALS.items():
        print(f"\nTrial {trial_num}: generating 10 samples...")
        gen = GPUSynthSweep(fine_np, coarse_np, N_LABELS, TARGET_SHAPE, device, cfg)
        trial_dir = out_base / f"trial_{trial_num}"

        for i in range(10):
            imgs, labs = gen.generate(batch_size=1)
            img_np = imgs[0, 0].cpu().numpy()
            lab_np = labs[0].cpu().numpy().astype(np.int32)
            save_grid(img_np, lab_np,
                      trial_dir / f"sample_{i:02d}.png",
                      title=f"Sweep Trial {trial_num} - Sample {i}")
            print(f"  sample_{i:02d}.png")
        del gen
        torch.cuda.empty_cache()

    del fine_np, coarse_np

    # ── 3. Real Abeta images ──
    print("\nReal Abeta images...")
    real_dir = out_base / "real"
    for sid in ["000", "005", "010"]:
        img_path = f"{BASE}/downsampled_lighsheet5/{sid}_SPIM.nii.gz"
        mask_path = f"{BASE}/brain_masks/{sid}_mask.nii.gz"
        if not Path(img_path).exists():
            print(f"  {sid}: not found, skipping")
            continue
        img = conform_volume(nib.load(img_path).get_fdata().astype(np.float32))
        mask = conform_volume(nib.load(mask_path).get_fdata().astype(np.float32))
        # Normalize image to [0,1]
        vmin, vmax = img.min(), img.max()
        if vmax - vmin > 1e-8:
            img = (img - vmin) / (vmax - vmin)
        save_grid(img, (mask > 0).astype(np.int32),
                  real_dir / f"real_{sid}.png",
                  title=f"Real Abeta - Subject {sid}")
        print(f"  real_{sid}.png")

    # ── 4. Experiment A baseline ──
    print("\nExperiment A baseline...")
    from utils.gpu_synth import GPUSynthGenerator
    label_paths = sorted(Path(f"{BASE}/labels").glob("*.nii*"))
    label_paths = [p for p in label_paths if p.name.split("_")[0] not in EXCLUDE]
    labels_np = [nib.load(str(p)).get_fdata().astype(np.int32) for p in label_paths]

    with open("configs/default.yaml", encoding="utf-8") as f:
        default_cfg = yaml.safe_load(f)

    gen_a = GPUSynthGenerator(labels_np, N_LABELS, TARGET_SHAPE, device,
                              default_cfg.get("synth", {}))
    del labels_np
    exp_a_dir = out_base / "experiment_a"

    for i in range(3):
        imgs, labs = gen_a.generate(batch_size=1)
        img_np = imgs[0, 0].cpu().numpy()
        lab_np = labs[0].cpu().numpy().astype(np.int32)
        save_grid(img_np, lab_np,
                  exp_a_dir / f"sample_{i:02d}.png",
                  title=f"Experiment A (baseline) - Sample {i}")
        print(f"  sample_{i:02d}.png")

    del gen_a
    torch.cuda.empty_cache()

    print(f"\nDone. All samples saved to {out_base}/")


if __name__ == "__main__":
    main()
