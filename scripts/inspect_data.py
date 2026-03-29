#!/usr/bin/env python3
"""
inspect_data.py - Quick inspection of SPIMquant outputs to determine
volume dimensions, number of labels, and data quality.

Usage:
    pixi run python scripts/inspect_data.py \
        --label-dir /nfs/khan/trainees/apooladi/brainhack/labels \
        --image-dir /nfs/khan/trainees/apooladi/brainhack/downsampled_lighsheet5
"""

import argparse
from pathlib import Path
import nibabel as nib
import numpy as np


def inspect(label_dir: str, image_dir: str = None):
    label_paths = sorted(Path(label_dir).glob("*.nii*"))
    print(f"Found {len(label_paths)} label map files\n")

    if not label_paths:
        print("No .nii/.nii.gz files found. Check your path.")
        return

    all_shapes = []
    all_labels = set()
    all_vox_sizes = []

    for i, p in enumerate(label_paths):
        nii = nib.load(str(p))
        shape = nii.shape[:3]
        vox = np.abs(np.diag(nii.affine[:3, :3]))
        labels = set(np.unique(nii.get_fdata().astype(np.int32)).tolist())

        all_shapes.append(shape)
        all_labels.update(labels)
        all_vox_sizes.append(tuple(np.round(vox, 3)))

        if i < 3 or i == len(label_paths) - 1:
            print(f"  {p.name}")
            print(f"    shape: {shape}  |  voxel size: {vox.round(3)}  |  unique labels: {len(labels)}")

    if len(label_paths) > 4:
        print(f"  ... ({len(label_paths) - 4} more)\n")

    shapes_arr = np.array(all_shapes)
    print(f"\n{'='*60}")
    print(f"SUMMARY (label maps)")
    print(f"{'='*60}")
    print(f"  Total files:     {len(label_paths)}")
    print(f"  Shape range:     min={tuple(shapes_arr.min(0))}  max={tuple(shapes_arr.max(0))}")
    print(f"  Shape median:    {tuple(np.median(shapes_arr, axis=0).astype(int))}")
    print(f"  Unique labels:   {len(all_labels)} (max index = {max(all_labels)})")
    print(f"  Voxel sizes:     {set(all_vox_sizes)}")

    # suggest target_shape
    median_shape = np.median(shapes_arr, axis=0).astype(int)
    suggested = tuple(int(np.ceil(s / 16) * 16) for s in median_shape)
    n_labels = max(all_labels) + 1

    print(f"\n  >>> CONFIG SUGGESTIONS <<<")
    print(f"  target_shape: [{suggested[0]}, {suggested[1]}, {suggested[2]}]")
    print(f"  n_labels: {n_labels}")

    if image_dir:
        img_paths = sorted(Path(image_dir).glob("*.nii*"))
        print(f"\n{'='*60}")
        print(f"IMAGES")
        print(f"{'='*60}")
        print(f"  Found {len(img_paths)} image files")

        if img_paths:
            nii = nib.load(str(img_paths[0]))
            print(f"  First image shape: {nii.shape}")
            print(f"  First image dtype: {nii.get_data_dtype()}")
            vol = nii.get_fdata()
            print(f"  Intensity range: [{vol.min():.1f}, {vol.max():.1f}]")
            print(f"  Mean: {vol.mean():.1f}, Std: {vol.std():.1f}")

        label_subs = {p.name.split("_")[0] for p in label_paths}
        image_subs = {p.name.split("_")[0] for p in img_paths}
        matched = label_subs & image_subs
        print(f"\n  Subjects with both label + image: {len(matched)}")

        unmatched_labels = label_subs - image_subs
        unmatched_images = image_subs - label_subs
        if unmatched_labels:
            print(f"  Labels without images: {unmatched_labels}")
        if unmatched_images:
            print(f"  Images without labels: {unmatched_images}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=str, required=True)
    parser.add_argument("--image-dir", type=str, default=None)
    args = parser.parse_args()
    inspect(args.label_dir, args.image_dir)
