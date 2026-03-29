#!/usr/bin/env python3
"""
train.py - SynthSeg-style two-phase training for SPIM brain segmentation.

Phase 1 (warmup): WeightedL2 loss on logits vs one-hot (first N steps)
Phase 2 (main):   Soft Dice loss only (remaining training)

Both phases: Adam optimizer, constant lr=1e-4, no weight decay, no scheduler.

Usage:
    pixi run python scripts/train.py --config configs/default.yaml
    pixi run python scripts/train.py --config configs/default.yaml --resume outputs/checkpoints/epoch_100.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unet import build_model, count_parameters
from models.losses import WeightedL2Loss, DiceLoss
from utils.synth_generator import SynthSegSPIMGenerator, RealImageDataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def discover_data(cfg: dict) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    """Discover label maps and images, split into train/val."""
    label_dir = Path(cfg["data"]["label_dir"])
    image_dir = Path(cfg["data"]["image_dir"])

    label_paths = sorted(label_dir.glob("*.nii*"))
    image_paths = sorted(image_dir.glob("*.nii*"))

    print(f"Found {len(label_paths)} label maps, {len(image_paths)} images")

    label_dict = {}
    for p in label_paths:
        sub = p.name.split("_")[0]
        label_dict[sub] = p

    image_dict = {}
    for p in image_paths:
        sub = p.name.split("_")[0]
        image_dict[sub] = p

    common_subs = sorted(set(label_dict.keys()) & set(image_dict.keys()))
    print(f"Matched {len(common_subs)} subjects with both label + image")

    rng = np.random.default_rng(cfg["data"]["random_seed"])
    rng.shuffle(common_subs)

    n_val = cfg["data"]["val_subjects"]
    val_subs = common_subs[:n_val]
    train_subs = common_subs[n_val:]

    print(f"Train: {len(train_subs)}, Val: {len(val_subs)}")

    train_labels = [label_dict[s] for s in train_subs]
    val_labels = [label_dict[s] for s in val_subs]
    val_images = [image_dict[s] for s in val_subs]

    return train_labels, val_labels, val_images, val_labels


def detect_n_labels(label_paths: list[Path]) -> int:
    """Scan all label maps to find the total number of unique labels."""
    import nibabel as nib
    all_labels = set()
    for p in label_paths:
        vol = nib.load(str(p)).get_fdata().astype(np.int32)
        all_labels.update(np.unique(vol).tolist())
    n = max(all_labels) + 1
    print(f"Detected {len(all_labels)} unique labels, using n_labels={n}")
    return n


def compute_dice_per_class(pred: torch.Tensor, target: torch.Tensor, n_classes: int) -> dict:
    """Compute Dice score per class for evaluation."""
    pred_labels = pred.argmax(dim=1)
    dices = {}
    for c in range(1, n_classes):  # skip background
        pred_c = (pred_labels == c).float()
        true_c = (target == c).float()
        if true_c.sum() == 0 and pred_c.sum() == 0:
            continue
        intersection = (pred_c * true_c).sum()
        dice = (2 * intersection) / (pred_c.sum() + true_c.sum() + 1e-8)
        dices[c] = dice.item()
    return dices


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_dices = []
    n_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()

        dices = compute_dice_per_class(logits, labels, n_classes)
        if dices:
            all_dices.append(np.mean(list(dices.values())))

        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_dice = np.mean(all_dices) if all_dices else 0.0

    return avg_loss, avg_dice


def save_checkpoint(model, optimizer, epoch, global_step, best_dice, path):
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_dice": best_dice,
        "model_state_dict": model.module.state_dict()
            if isinstance(model, nn.DataParallel) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser(description="Train SynthSegSPIM")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- output dirs ----
    out_dir = Path(cfg["data"]["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # ---- discover data ----
    train_label_paths, val_label_paths, val_image_paths, _ = discover_data(cfg)

    if cfg["volume"]["n_labels"] == 0:
        all_paths = train_label_paths + val_label_paths
        cfg["volume"]["n_labels"] = detect_n_labels(all_paths)

    n_labels = cfg["volume"]["n_labels"]
    target_shape = tuple(cfg["volume"]["target_shape"])
    train_cfg = cfg["training"]
    warmup_steps = train_cfg["warmup_steps"]

    print(f"\n{'='*60}")
    print(f"SynthSegSPIM Training (two-phase)")
    print(f"  n_labels:      {n_labels}")
    print(f"  target_shape:  {target_shape}")
    print(f"  train maps:    {len(train_label_paths)}")
    print(f"  val subjects:  {len(val_image_paths)}")
    print(f"  warmup_steps:  {warmup_steps} (phase 1: WL2)")
    print(f"  total epochs:  {train_cfg['epochs']} x {train_cfg['steps_per_epoch']} steps/epoch")
    print(f"{'='*60}\n")

    # ---- datasets ----
    train_dataset = SynthSegSPIMGenerator(
        label_paths=train_label_paths,
        n_labels=n_labels,
        target_shape=target_shape,
        samples_per_epoch=train_cfg["steps_per_epoch"],
        cfg_synth=cfg.get("synth", {}),
    )

    val_dataset = RealImageDataset(
        image_paths=val_image_paths,
        label_paths=val_label_paths,
        target_shape=target_shape,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ---- model ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)

    gpu_ids = train_cfg.get("gpu_ids", [0])
    if len(gpu_ids) > 1 and torch.cuda.device_count() >= len(gpu_ids):
        print(f"Using DataParallel on GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)

    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # ---- two-phase losses ----
    wl2_loss = WeightedL2Loss(n_labels).to(device)
    dice_loss = DiceLoss(n_labels).to(device)

    # ---- optimizer: Adam, constant LR, no weight decay ----
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg["amp"])
    use_amp = train_cfg["amp"]

    # ---- resume ----
    start_epoch = 0
    global_step = 0
    best_dice = 0.0

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", start_epoch * train_cfg["steps_per_epoch"])
        best_dice = ckpt.get("best_dice", 0.0)
        print(f"  Resumed at epoch {start_epoch}, step {global_step}, best_dice={best_dice:.4f}")

    log_path = log_dir / "training_log.jsonl"

    # ---- training loop ----
    print(f"\nStarting training for {train_cfg['epochs']} epochs...")
    print(f"  Phase 1 (WL2):  steps 0 - {warmup_steps}")
    print(f"  Phase 2 (Dice): steps {warmup_steps}+")
    print(f"  Optimizer: Adam, lr={train_cfg['lr']}, no weight decay")
    print(f"  AMP: {use_amp}")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print()

    for epoch in range(start_epoch, train_cfg["epochs"]):
        t0 = time.time()
        model.train()

        epoch_losses = []
        phase_at_start = "wl2" if global_step < warmup_steps else "dice"

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Select loss based on phase
            criterion = wl2_loss if global_step < warmup_steps else dice_loss

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1

            # Log phase transition
            if global_step == warmup_steps:
                print(f"  >>> Phase transition at step {global_step}: WL2 -> Dice <<<")

        avg_train_loss = np.mean(epoch_losses)
        elapsed = time.time() - t0
        current_phase = "wl2" if global_step < warmup_steps else "dice"
        current_lr = optimizer.param_groups[0]["lr"]

        log_entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(avg_train_loss),
            "phase": current_phase,
            "lr": current_lr,
            "elapsed_s": elapsed,
        }

        # ---- validate ----
        if (epoch + 1) % train_cfg["val_every"] == 0:
            val_loss, val_dice = validate(
                model, val_loader, dice_loss, device, n_labels, use_amp
            )

            log_entry["val_loss"] = float(val_loss)
            log_entry["val_dice"] = float(val_dice)

            print(
                f"Epoch {epoch:4d}/{train_cfg['epochs']} | "
                f"step={global_step} | "
                f"phase={current_phase} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_dice={val_dice:.4f} | "
                f"{elapsed:.1f}s"
            )

            if val_dice > best_dice:
                best_dice = val_dice
                save_checkpoint(model, optimizer, epoch, global_step, best_dice,
                                ckpt_dir / "best.pt")
                print(f"  >> New best dice: {best_dice:.4f}")
        else:
            print(
                f"Epoch {epoch:4d}/{train_cfg['epochs']} | "
                f"step={global_step} | "
                f"phase={current_phase} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"{elapsed:.1f}s"
            )

        # ---- periodic checkpoint ----
        if (epoch + 1) % train_cfg["save_every"] == 0:
            save_checkpoint(
                model, optimizer, epoch, global_step, best_dice,
                ckpt_dir / f"epoch_{epoch+1:04d}.pt"
            )

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    # ---- save final ----
    save_checkpoint(model, optimizer, train_cfg["epochs"] - 1, global_step, best_dice,
                    ckpt_dir / "final.pt")
    print(f"\nTraining complete. Best val dice: {best_dice:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
