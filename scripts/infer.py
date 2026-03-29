#!/usr/bin/env python3
"""
infer.py - Run trained SynthSegSPIM model on new subjects.

Takes a level-5 downsampled NIfTI, outputs a dseg NIfTI in subject space.

Usage:
    pixi run python scripts/infer.py \
        --checkpoint outputs/checkpoints/best.pt \
        --config outputs/config.yaml \
        --input /path/to/sub-001_level-5_SPIM.nii.gz \
        --output /path/to/sub-001_dseg.nii.gz

    # batch mode
    pixi run python scripts/infer.py \
        --checkpoint outputs/checkpoints/best.pt \
        --config outputs/config.yaml \
        --input-dir /path/to/level5_niis/ \
        --output-dir /path/to/predicted_dsegs/
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unet import build_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def conform_volume(vol: np.ndarray, target_shape: tuple) -> tuple[np.ndarray, dict]:
    """Pad/crop to target shape, return the volume and metadata needed to undo it."""
    D, H, W = target_shape
    orig_shape = vol.shape[:3]
    out = np.zeros((D, H, W), dtype=vol.dtype)

    sd, sh, sw = orig_shape
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

    meta = {
        "orig_shape": orig_shape,
        "crop_start": (d0, h0, w0),
        "pad_start": (od, oh, ow),
        "copy_size": (copy_d, copy_h, copy_w),
    }
    return out, meta


def unconform_volume(vol: np.ndarray, meta: dict) -> np.ndarray:
    """Undo conform: extract the original-shaped region from the conformed volume."""
    orig_shape = meta["orig_shape"]
    out = np.zeros(orig_shape, dtype=vol.dtype)

    od, oh, ow = meta["pad_start"]
    copy_d, copy_h, copy_w = meta["copy_size"]
    d0, h0, w0 = meta["crop_start"]

    out[d0:d0+copy_d, h0:h0+copy_h, w0:w0+copy_w] = \
        vol[od:od+copy_d, oh:oh+copy_h, ow:ow+copy_w]

    return out


def predict_single(
    model: torch.nn.Module,
    nii_path: Path,
    target_shape: tuple,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Run inference on a single NIfTI volume.
    Returns predicted label map in original space + reference NIfTI for header/affine."""
    nii = nib.load(str(nii_path))
    vol = nii.get_fdata().astype(np.float32)

    # conform
    vol_conf, meta = conform_volume(vol, target_shape)

    # normalize
    vol_norm = (vol_conf - vol_conf.min()) / (vol_conf.max() - vol_conf.min() + 1e-8)

    # to tensor
    x = torch.from_numpy(vol_norm[np.newaxis, np.newaxis]).float().to(device)

    # predict
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)

    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

    # unconform back to original shape
    pred_orig = unconform_volume(pred, meta)

    return pred_orig, nii


def main():
    parser = argparse.ArgumentParser(description="SynthSegSPIM Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, default=None, help="Single NIfTI input")
    parser.add_argument("--output", type=str, default=None, help="Single NIfTI output")
    parser.add_argument("--input-dir", type=str, default=None, help="Batch input dir")
    parser.add_argument("--output-dir", type=str, default=None, help="Batch output dir")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_shape = tuple(cfg["volume"]["target_shape"])

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # load model
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {args.checkpoint} (epoch {ckpt['epoch']}, dice {ckpt.get('best_dice', '?')})")

    # single mode
    if args.input and args.output:
        pred, ref_nii = predict_single(model, Path(args.input), target_shape, device)
        out_nii = nib.Nifti1Image(pred, ref_nii.affine, ref_nii.header)
        nib.save(out_nii, args.output)
        print(f"Saved: {args.output}")
        return

    # batch mode
    if args.input_dir and args.output_dir:
        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        nii_files = sorted(in_dir.glob("*.nii*"))
        print(f"Processing {len(nii_files)} volumes...")

        for nii_path in nii_files:
            pred, ref_nii = predict_single(model, nii_path, target_shape, device)

            out_name = nii_path.name.replace(".nii.gz", "_dseg.nii.gz").replace(".nii", "_dseg.nii")
            out_path = out_dir / out_name

            out_nii = nib.Nifti1Image(pred, ref_nii.affine, ref_nii.header)
            nib.save(out_nii, str(out_path))
            print(f"  {nii_path.name} -> {out_name}")

        print("Done.")
        return

    print("Provide either --input/--output or --input-dir/--output-dir")


if __name__ == "__main__":
    main()
