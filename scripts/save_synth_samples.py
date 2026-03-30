#!/usr/bin/env python3
"""Save synthetic samples as NIfTI files for visual inspection.

Usage:
    pixi run python scripts/save_synth_samples.py --config configs/default.yaml --gpu 0 --n 20
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.gpu_synth import GPUSynthGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--output-dir", default="/nfs/khan/trainees/apooladi/brainhack/generated_synth_test")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    target_shape = tuple(cfg["volume"]["target_shape"])
    n_labels = cfg["volume"]["n_labels"]

    # Load label maps
    from pathlib import Path as P
    label_dir = P(cfg["data"]["label_dir"])
    label_paths = sorted(label_dir.glob("*.nii*"))
    # Use same exclusions as training
    exclude = {"022", "032", "041"}
    label_paths = [p for p in label_paths if p.name.split("_")[0] not in exclude]

    print(f"Loading {len(label_paths)} label maps...")
    label_maps_np = []
    for p in label_paths:
        label_maps_np.append(nib.load(str(p)).get_fdata().astype(np.int32))

    gen = GPUSynthGenerator(
        label_maps_np=label_maps_np,
        n_labels=n_labels,
        target_shape=target_shape,
        device=device,
        cfg=cfg.get("synth", {}),
    )
    del label_maps_np

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use a simple identity affine with the actual voxel sizes
    vox = cfg["volume"].get("voxel_size", [0.052, 0.052, 0.035216])
    affine = np.diag([*vox, 1.0]) if vox else np.eye(4)

    print(f"Generating {args.n} synthetic samples to {out_dir}/")
    for i in range(args.n):
        img, lab = gen.generate(batch_size=1)
        img_np = img[0, 0].cpu().numpy().astype(np.float32)  # (D, H, W)
        lab_np = lab[0].cpu().numpy().astype(np.int32)        # (D, H, W)

        nib.save(nib.Nifti1Image(img_np, affine), str(out_dir / f"synth_{i:03d}_image.nii.gz"))
        nib.save(nib.Nifti1Image(lab_np.astype(np.float32), affine), str(out_dir / f"synth_{i:03d}_label.nii.gz"))
        print(f"  [{i+1}/{args.n}] range=[{img_np.min():.3f}, {img_np.max():.3f}], "
              f"labels={len(np.unique(lab_np))}, brain={100*(lab_np>0).mean():.1f}%")

    print(f"\nDone. Open in ITK-SNAP or fsleyes:")
    print(f"  fsleyes {out_dir}/synth_000_image.nii.gz {out_dir}/synth_000_label.nii.gz -cm random")


if __name__ == "__main__":
    main()
