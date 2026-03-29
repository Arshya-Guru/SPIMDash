#!/usr/bin/env python3
"""
train.py — Plain PyTorch single-GPU training with GPU-native synthetic data generation.

Two-phase SynthSeg training:
  Phase 1 (steps 0 to warmup_steps): Weighted L2 loss
  Phase 2 (steps warmup_steps+):     Soft Dice loss

Optimizer: Adam, constant lr=1e-4, no weight decay.
Data generation: entirely on GPU via GPUSynthGenerator (~30ms/sample).

Usage:
    pixi run python scripts/train.py --config configs/default.yaml
    pixi run python scripts/train.py --config configs/default.yaml --resume outputs/checkpoints/best.pt
"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unet import build_model, count_parameters
from models.losses import DiceCELoss, DiceLoss
from utils.gpu_synth import GPUSynthGenerator
from utils.synth_generator import RealImageDataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def discover_data(cfg: dict):
    """Discover and split data. Returns train_label_paths, val_label_paths, val_image_paths."""
    label_dir = Path(cfg["data"]["label_dir"])
    image_dir = Path(cfg["data"]["image_dir"])

    label_paths = sorted(label_dir.glob("*.nii*"))
    image_paths = sorted(image_dir.glob("*.nii*"))

    label_dict = {p.name.split("_")[0]: p for p in label_paths}
    image_dict = {p.name.split("_")[0]: p for p in image_paths}

    common = sorted(set(label_dict) & set(image_dict))
    exclude_subjects = {"022", "032", "041"}
    excluded = [s for s in common if s in exclude_subjects]
    common = [s for s in common if s not in exclude_subjects]
    if excluded:
        print(f"Excluded {len(excluded)} bad subjects: {excluded}")
    rng = np.random.default_rng(cfg["data"]["random_seed"])
    rng.shuffle(common)

    n_val = cfg["data"]["val_subjects"]
    train_subs = common[n_val:]
    val_subs = common[:n_val]

    print(f"Data: {len(common)} subjects, {len(train_subs)} train, {len(val_subs)} val")

    return (
        [label_dict[s] for s in train_subs],
        [label_dict[s] for s in val_subs],
        [image_dict[s] for s in val_subs],
    )


def compute_dice_per_class(logits, targets, n_classes):
    """Per-class Dice for validation."""
    pred = logits.argmax(dim=1)
    dices = {}
    for c in range(1, n_classes):
        pc = (pred == c).float()
        tc = (targets == c).float()
        if tc.sum() == 0 and pc.sum() == 0:
            continue
        dice = (2 * (pc * tc).sum()) / (pc.sum() + tc.sum() + 1e-8)
        dices[c] = dice.item()
    return dices


@torch.no_grad()
def validate(model, val_dataset, device, dice_loss, n_labels, use_amp):
    """Run validation on real image+label pairs."""
    model.eval()
    total_loss = 0
    all_dices = []

    for i in range(len(val_dataset)):
        img_t, lab_t = val_dataset[i]
        img_t = img_t.unsqueeze(0).to(device)  # (1, 1, D, H, W)
        lab_t = lab_t.unsqueeze(0).to(device)   # (1, D, H, W)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(img_t)
            loss = dice_loss(logits, lab_t)

        total_loss += loss.item()
        dices = compute_dice_per_class(logits, lab_t, n_labels)
        if dices:
            all_dices.append(np.mean(list(dices.values())))

    avg_loss = total_loss / max(len(val_dataset), 1)
    avg_dice = np.mean(all_dices) if all_dices else 0.0
    model.train()
    return avg_loss, avg_dice


def save_checkpoint(model, optimizer, epoch, global_step, best_dice, path):
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "best_dice": best_dice,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index (default 0, use 1 for parallel experiment)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]

    # TF32 for A100 tensor cores
    torch.set_float32_matmul_precision("medium")

    device = torch.device(f"cuda:{args.gpu}")
    print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(device)}")

    # Output dirs
    out_dir = Path(cfg["data"]["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # Discover data
    train_label_paths, val_label_paths, val_image_paths = discover_data(cfg)

    n_labels = cfg["volume"]["n_labels"]
    target_shape = tuple(cfg["volume"]["target_shape"])
    steps_per_epoch = train_cfg["steps_per_epoch"]
    use_amp = train_cfg["amp"]

    # Load label maps as numpy (native sizes)
    print("Loading training label maps...")
    train_labels_np = []
    for p in train_label_paths:
        train_labels_np.append(nib.load(str(p)).get_fdata().astype(np.int32))

    # Build GPU synth generator
    gpu_synth = GPUSynthGenerator(
        label_maps_np=train_labels_np,
        n_labels=n_labels,
        target_shape=target_shape,
        device=device,
        cfg=cfg.get("synth", {}),
    )
    del train_labels_np  # free numpy copies

    # Validation dataset (preloaded on CPU, moved to GPU on demand)
    val_dataset = RealImageDataset(
        image_paths=val_image_paths,
        label_paths=val_label_paths,
        target_shape=target_shape,
    )

    # Model
    model = build_model(cfg).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Loss: Dice+CE from step 0
    criterion = DiceCELoss(n_labels).to(device)
    dice_loss = DiceLoss(n_labels).to(device)  # for validation metric

    # Optimizer: Adam, constant lr, no weight decay (SynthSeg protocol)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Resume
    start_epoch = 0
    global_step = 0
    best_dice = 0.0

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"]
        # Handle Lightning-style keys if resuming from old checkpoints
        if any(k.startswith("model.") for k in state_dict):
            state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()
                          if k.startswith("model.")}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", start_epoch * steps_per_epoch)
        best_dice = ckpt.get("best_dice", 0.0)
        print(f"  Resumed at epoch {start_epoch}, step {global_step}, dice {best_dice:.4f}")

    # Logging
    log_path = log_dir / "training_log.jsonl"

    # Print training plan
    total_steps = train_cfg["epochs"] * steps_per_epoch
    print(f"\n{'='*60}")
    print(f"SynthSegSPIM Training (single GPU, GPU-native synth)")
    print(f"  n_labels:      {n_labels}")
    print(f"  target_shape:  {target_shape}")
    print(f"  train maps:    {len(gpu_synth.label_maps)} (on GPU)")
    print(f"  val subjects:  {len(val_dataset)}")
    print(f"  Loss:          Dice+CE from step 0")
    print(f"  Total steps:   {total_steps} ({train_cfg['epochs']} epochs x {steps_per_epoch})")
    print(f"  Optimizer:     Adam, lr={train_cfg['lr']}, constant")
    print(f"  AMP:           {use_amp}")
    print(f"  Batch size:    {train_cfg['batch_size']}")
    print(f"{'='*60}\n")

    # ==================== TRAINING LOOP ====================
    model.train()

    for epoch in range(start_epoch, train_cfg["epochs"]):
        t0 = time.time()
        epoch_losses = []

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}", leave=False)
        for step in pbar:
            # Generate synthetic data ON GPU (~30ms)
            images, labels = gpu_synth.generate(batch_size=train_cfg["batch_size"])
            torch.cuda.empty_cache()  # reclaim fragmented reserved memory

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

        elapsed = time.time() - t0
        avg_loss = np.mean(epoch_losses)

        log_entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(avg_loss),
            "elapsed_s": elapsed,
            "steps_per_sec": steps_per_epoch / elapsed,
        }

        # Validate
        if (epoch + 1) % train_cfg["val_every"] == 0:
            val_loss, val_dice = validate(
                model, val_dataset, device, dice_loss, n_labels, use_amp,
            )
            log_entry["val_loss"] = float(val_loss)
            log_entry["val_dice"] = float(val_dice)

            print(f"Epoch {epoch:4d} | step {global_step} | "
                  f"train={avg_loss:.4f} val={val_loss:.4f} dice={val_dice:.4f} | "
                  f"{elapsed:.1f}s ({steps_per_epoch / elapsed:.1f} steps/s)")

            if val_dice > best_dice:
                best_dice = val_dice
                save_checkpoint(model, optimizer, epoch, global_step, best_dice,
                                ckpt_dir / "best.pt")
                print(f"  >> New best dice: {best_dice:.4f}")
        else:
            print(f"Epoch {epoch:4d} | step {global_step} | "
                  f"train={avg_loss:.4f} | {elapsed:.1f}s ({steps_per_epoch / elapsed:.1f} steps/s)")

        # Periodic checkpoint
        if (epoch + 1) % train_cfg["save_every"] == 0:
            save_checkpoint(model, optimizer, epoch, global_step, best_dice,
                            ckpt_dir / f"epoch_{epoch+1:04d}.pt")

        # Log
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    # Final save
    save_checkpoint(model, optimizer, train_cfg["epochs"] - 1, global_step, best_dice,
                    ckpt_dir / "final.pt")
    print(f"\nDone. Best val dice: {best_dice:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
