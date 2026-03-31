#!/usr/bin/env python3
"""
train_sweep.py — Training with sweep-optimized synthetic generation (Experiments G/H).

Uses GPUSynthSweep generator with Trial 257 params. Auto-resumes from latest checkpoint.

Usage:
    pixi run python scripts/train_sweep.py --config configs/experiment_g.yaml --gpu 0
    pixi run python scripts/train_sweep.py --config configs/experiment_h.yaml --gpu 1
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unet import SynthSegUNet, count_parameters
from models.losses import DiceCELoss, DiceLoss
from utils.gpu_synth_sweep import GPUSynthSweep
from utils.synth_generator import conform_volume


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_split(cfg):
    """Determine train/val split — MUST match train.py exactly."""
    coarse_dir = Path(cfg["data"]["coarse_label_dir"])
    image_dir = Path(cfg["data"]["val_image_dir"])
    fine_dir = Path(cfg["data"]["fine_label_dir"])

    coarse_dict = {p.name.split("_")[0]: p for p in sorted(coarse_dir.glob("*.nii*"))}
    image_dict = {p.name.split("_")[0]: p for p in sorted(image_dir.glob("*.nii*"))}
    fine_dict = {p.name.split("_")[0]: p for p in sorted(fine_dir.glob("*.nii*"))}

    # Same logic as train.py: intersection of coarse labels + images
    common = sorted(set(coarse_dict) & set(image_dict))
    exclude = set(cfg["data"]["exclude_subjects"])
    common = [s for s in common if s not in exclude]

    # Same seed, same shuffle, same split
    rng = np.random.default_rng(cfg["data"]["random_seed"])
    rng.shuffle(common)

    n_val = cfg["data"]["val_subjects"]
    val_subs = common[:n_val]
    train_subs = common[n_val:]

    # Filter train to subjects that also have fine labels
    train_subs = [s for s in train_subs if s in fine_dict]

    return train_subs, val_subs, coarse_dict, fine_dict, image_dict


def load_val_data(val_subs, image_dict, coarse_dict, target_shape, olf_labels):
    """Load validation images + labels into memory."""
    val_images = []
    val_labels = []
    for sid in val_subs:
        img = nib.load(str(image_dict[sid])).get_fdata().astype(np.float32)
        lab = nib.load(str(coarse_dict[sid])).get_fdata().astype(np.int32)
        img = conform_volume(img, target_shape)
        lab = conform_volume(lab, target_shape)
        # Merge olfactory
        for ol in olf_labels:
            lab[lab == ol] = 0
        # Normalize image
        vmin, vmax = img.min(), img.max()
        if vmax - vmin > 1e-8:
            img = (img - vmin) / (vmax - vmin)
        val_images.append(torch.from_numpy(img[np.newaxis]).float())  # (1, D, H, W)
        val_labels.append(torch.from_numpy(lab).long())                # (D, H, W)
    return val_images, val_labels


@torch.no_grad()
def validate(model, val_images, val_labels, device, n_labels, use_amp):
    """Run validation, return mean foreground Dice + per-label dict."""
    model.eval()
    all_dices = {}

    for img_t, lab_t in zip(val_images, val_labels):
        img_t = img_t.unsqueeze(0).to(device)  # (1, 1, D, H, W)
        lab_t = lab_t.unsqueeze(0).to(device)   # (1, D, H, W)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(img_t)

        pred = logits.argmax(dim=1)
        for c in range(1, n_labels):
            pc = (pred == c).float()
            tc = (lab_t == c).float()
            if tc.sum() == 0 and pc.sum() == 0:
                continue
            dice = (2 * (pc * tc).sum() / (pc.sum() + tc.sum() + 1e-8)).item()
            if c not in all_dices:
                all_dices[c] = []
            all_dices[c].append(dice)

    per_label = {str(c): float(np.mean(v)) for c, v in sorted(all_dices.items())}
    mean_dice = float(np.mean(list(per_label.values()))) if per_label else 0.0
    model.train()
    return mean_dice, per_label


def save_checkpoint(path, epoch, global_step, model, optimizer, scaler,
                    best_dice, val_subject_ids, config_path):
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_dice": best_dice,
        "val_subject_ids": val_subject_ids,
        "config_path": config_path,
    }, path)


def cleanup_old_checkpoints(ckpt_dir, keep=3):
    """Keep only the last N epoch_*.pt checkpoints."""
    epoch_files = sorted(Path(ckpt_dir).glob("epoch_*.pt"))
    for f in epoch_files[:-keep]:
        f.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    synth_cfg = cfg["synth"]
    out_cfg = cfg["output"]
    target_shape = tuple(cfg["volume"]["target_shape"])
    n_labels = cfg["volume"]["n_labels"]
    olf_labels = synth_cfg.get("olfactory_labels", [1, 2])

    # Device
    device = torch.device(f"cuda:{args.gpu}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    print(f"GPU {args.gpu}: {torch.cuda.get_device_name(device)}")
    print(f"Config: {args.config}")

    # Output dirs
    ckpt_dir = Path(out_cfg["checkpoint_dir"])
    log_dir = Path(out_cfg["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(out_cfg["log_file"])

    # Train/val split
    train_subs, val_subs, coarse_dict, fine_dict, image_dict = discover_split(cfg)
    print(f"Train subjects ({len(train_subs)}): {sorted(train_subs)}")
    print(f"Val subjects ({len(val_subs)}): {sorted(val_subs)}")

    # Load data
    print("Loading fine label maps...")
    fine_np = [nib.load(str(fine_dict[s])).get_fdata().astype(np.int32) for s in train_subs]
    print("Loading coarse label maps...")
    coarse_np = [nib.load(str(coarse_dict[s])).get_fdata().astype(np.int32) for s in train_subs]
    print("Loading validation data...")
    val_images, val_labels = load_val_data(val_subs, image_dict, coarse_dict,
                                           target_shape, olf_labels)
    print(f"  {len(val_images)} val pairs loaded")

    # Generator
    print("Building GPU synth generator...")
    gen = GPUSynthSweep(fine_np, coarse_np, n_labels, target_shape, device, cfg=synth_cfg)
    del fine_np, coarse_np

    # Model
    model = SynthSegUNet(
        in_channels=cfg["model"].get("in_channels", 1),
        n_labels=n_labels,
        base_features=cfg["model"].get("base_features", 24),
    ).to(device)
    print(f"Model: {count_parameters(model):,} params")

    # Loss + optimizer
    criterion = DiceCELoss(n_labels).to(device)
    dice_loss_fn = DiceLoss(n_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg["amp"])
    use_amp = train_cfg["amp"]

    # Resume logic
    start_epoch = 0
    global_step = 0
    best_dice = 0.0
    latest_path = ckpt_dir / "latest.pt"

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_dice = ckpt["best_dice"]
        # Verify split
        if ckpt.get("val_subject_ids") and sorted(ckpt["val_subject_ids"]) != sorted(val_subs):
            print(f"ERROR: Val split mismatch! Checkpoint: {sorted(ckpt['val_subject_ids'])}")
            print(f"       Current: {sorted(val_subs)}")
            sys.exit(1)
        print(f"Resumed from {args.resume}: epoch {start_epoch}, step {global_step}, dice {best_dice:.4f}")
    elif not args.fresh and latest_path.exists():
        print(f"WARNING: Found existing checkpoint at {latest_path}. Auto-resuming.")
        print(f"Use --fresh to force a clean start.")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_dice = ckpt["best_dice"]
        if ckpt.get("val_subject_ids") and sorted(ckpt["val_subject_ids"]) != sorted(val_subs):
            print(f"ERROR: Val split mismatch!")
            sys.exit(1)
        print(f"Auto-resumed: epoch {start_epoch}, step {global_step}, dice {best_dice:.4f}")

    # Print summary
    n_epochs = train_cfg["n_epochs"]
    steps_per_epoch = train_cfg["steps_per_epoch"]
    batch_size = train_cfg["batch_size"]
    total_steps = n_epochs * steps_per_epoch

    print(f"\n{'='*60}")
    print(f"Training: {args.config}")
    print(f"  Epochs: {start_epoch} -> {n_epochs} ({n_epochs - start_epoch} remaining)")
    print(f"  Steps/epoch: {steps_per_epoch}, batch: {batch_size}")
    print(f"  Total steps: {total_steps}")
    print(f"  Loss: Dice+CE, lr: {train_cfg['lr']}, AMP: {use_amp}")
    print(f"  GPU mem: {torch.cuda.memory_allocated(device)/1e9:.1f} GB allocated")
    print(f"{'='*60}\n")

    # ==================== TRAINING LOOP ====================
    model.train()

    for epoch in range(start_epoch, n_epochs):
        t_epoch = time.time()
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            images, labels = gen.generate(batch_size=batch_size)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            epoch_loss += loss.item()

            if (step + 1) % 100 == 0:
                entry = {
                    "type": "train", "epoch": epoch + 1,
                    "step": step + 1, "global_step": global_step,
                    "loss": round(loss.item(), 5),
                    "lr": train_cfg["lr"],
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

                if (step + 1) % 500 == 0:
                    print(f"  Epoch {epoch+1} step {step+1}/{steps_per_epoch} | "
                          f"loss={loss.item():.4f} | "
                          f"{(step+1)/(time.time()-t_epoch):.2f} steps/s")

        # Epoch summary
        avg_loss = epoch_loss / steps_per_epoch
        elapsed = time.time() - t_epoch
        steps_sec = steps_per_epoch / elapsed
        remaining = n_epochs - epoch - 1
        eta_hours = remaining * elapsed / 3600

        print(f"Epoch {epoch+1}/{n_epochs} | loss={avg_loss:.4f} | "
              f"{steps_sec:.2f} steps/s | {elapsed/60:.1f} min/epoch | "
              f"ETA: {eta_hours:.1f}h")

        entry = {
            "type": "epoch", "epoch": epoch + 1,
            "global_step": global_step,
            "avg_loss": round(avg_loss, 5),
            "steps_sec": round(steps_sec, 2),
            "elapsed_min": round(elapsed / 60, 1),
            "eta_hours": round(eta_hours, 1),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Validation
        if (epoch + 1) % train_cfg["val_every_n_epochs"] == 0:
            val_dice, per_label = validate(
                model, val_images, val_labels, device, n_labels, use_amp)
            print(f"  Val Dice: {val_dice:.4f} (best: {best_dice:.4f})")

            val_entry = {
                "type": "val", "epoch": epoch + 1,
                "global_step": global_step,
                "mean_dice": round(val_dice, 5),
                "per_label_dice": per_label,
                "best_dice": round(max(best_dice, val_dice), 5),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(val_entry) + "\n")

            if val_dice > best_dice:
                best_dice = val_dice
                save_checkpoint(ckpt_dir / "best.pt", epoch, global_step,
                                model, optimizer, scaler, best_dice,
                                val_subs, args.config)
                print(f"  New best! Saved best.pt")

        # Periodic checkpoint
        if (epoch + 1) % train_cfg["save_every_n_epochs"] == 0:
            save_checkpoint(latest_path, epoch, global_step,
                            model, optimizer, scaler, best_dice,
                            val_subs, args.config)
            save_checkpoint(ckpt_dir / f"epoch_{epoch+1:04d}.pt", epoch,
                            global_step, model, optimizer, scaler, best_dice,
                            val_subs, args.config)
            cleanup_old_checkpoints(ckpt_dir, keep=train_cfg["keep_last_n_checkpoints"])

    # Final save
    save_checkpoint(ckpt_dir / "final.pt", n_epochs - 1, global_step,
                    model, optimizer, scaler, best_dice, val_subs, args.config)
    print(f"\nDone. Best val dice: {best_dice:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
