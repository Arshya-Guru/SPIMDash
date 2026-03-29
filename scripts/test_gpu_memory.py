#!/usr/bin/env python3
"""
test_gpu_memory.py - Quick GPU memory test for the U-Net at target_shape.

Usage:
    pixi run python scripts/test_gpu_memory.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unet import build_model, count_parameters
from models.losses import build_loss


def test_memory(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")
    else:
        print("  No CUDA available - running on CPU (memory test only meaningful on GPU)")

    vol_cfg = cfg["volume"]
    train_cfg = cfg["training"]
    target_shape = tuple(vol_cfg["target_shape"])
    n_labels = vol_cfg["n_labels"]
    batch_size = train_cfg["batch_size"]
    use_amp = train_cfg["amp"] and torch.cuda.is_available()

    print(f"\nTarget shape: {target_shape}")
    print(f"Batch size: {batch_size}")
    print(f"N labels: {n_labels}")
    print(f"AMP: {use_amp}")

    # Build model
    model = build_model(cfg)
    print(f"Model parameters: {count_parameters(model):,}")

    gpu_ids = train_cfg.get("gpu_ids", [0])
    if len(gpu_ids) > 1 and torch.cuda.device_count() >= len(gpu_ids):
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)

    criterion = build_loss(cfg).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Create dummy data
    dummy_img = torch.randn(batch_size, 1, *target_shape, device=device)
    dummy_lab = torch.randint(0, n_labels, (batch_size, *target_shape), device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Forward pass
    print("\nRunning forward pass...")
    with torch.amp.autocast("cuda", enabled=use_amp):
        logits = model(dummy_img)
        loss = criterion(logits, dummy_lab)
    print(f"  Output shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    if torch.cuda.is_available():
        fwd_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory after forward: {fwd_mem:.2f} GB")

    # Backward pass
    print("Running backward pass...")
    scaler.scale(loss).backward()

    if torch.cuda.is_available():
        bwd_mem = torch.cuda.max_memory_allocated() / 1e9
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Peak GPU memory after backward: {bwd_mem:.2f} GB")
        print(f"  Total GPU memory: {total_mem:.1f} GB")
        print(f"  Usage: {bwd_mem/total_mem*100:.1f}%")

        if bwd_mem < total_mem * 0.85:
            print("\n  >>> FITS in GPU memory <<<")
            if batch_size == 1:
                print("\n  Testing batch_size=2...")
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    model.zero_grad(set_to_none=True)
                    d2 = torch.randn(2, 1, *target_shape, device=device)
                    l2 = torch.randint(0, n_labels, (2, *target_shape), device=device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        o2 = model(d2)
                        loss2 = criterion(o2, l2)
                    scaler.scale(loss2).backward()
                    bs2_mem = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  Peak with batch_size=2: {bs2_mem:.2f} GB ({bs2_mem/total_mem*100:.1f}%)")
                    if bs2_mem < total_mem * 0.85:
                        print("  >>> batch_size=2 FITS - update config <<<")
                    else:
                        print("  >>> batch_size=2 too tight, keep batch_size=1 <<<")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  batch_size=2 OOM - keep batch_size=1")
                    else:
                        raise
        else:
            print("\n  >>> TIGHT - consider reducing model size or using patches <<<")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    test_memory(cfg)
