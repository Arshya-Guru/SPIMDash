#!/usr/bin/env python3
"""Experiment E: roi198 fine parcellation + full SynthSeg-faithful augmentation."""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unet import build_model, count_parameters
from models.losses import DiceCELoss, DiceLoss
from utils.gpu_synth_fine_v2 import GPUSynthGeneratorFineV2
from utils.synth_generator import RealImageDataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def discover_data(cfg: dict):
    coarse_dir = Path(cfg["data"]["label_dir"])
    fine_dir = Path(cfg["data"]["fine_label_dir"])
    image_dir = Path(cfg["data"]["image_dir"])

    coarse_dict = {p.name.split("_")[0]: p for p in sorted(coarse_dir.glob("*.nii*"))}
    fine_dict = {p.name.split("_")[0]: p for p in sorted(fine_dir.glob("*.nii*"))}
    image_dict = {p.name.split("_")[0]: p for p in sorted(image_dir.glob("*.nii*"))}

    common = sorted(set(coarse_dict) & set(fine_dict) & set(image_dict))
    exclude_subjects = {"022", "032", "041"}
    excluded = [s for s in common if s in exclude_subjects]
    common = [s for s in common if s not in exclude_subjects]
    if excluded:
        print(f"Excluded {len(excluded)} bad subjects: {excluded}")

    rng = np.random.default_rng(cfg["data"]["random_seed"])
    rng.shuffle(common)

    n_val = cfg["data"]["val_subjects"]
    train_subs, val_subs = common[n_val:], common[:n_val]
    print(f"Data: {len(common)} subjects, {len(train_subs)} train, {len(val_subs)} val")

    return (
        [fine_dict[s] for s in train_subs],
        [coarse_dict[s] for s in train_subs],
        [coarse_dict[s] for s in val_subs],
        [image_dict[s] for s in val_subs],
    )


def compute_dice_per_class(logits, targets, n_classes):
    pred = logits.argmax(dim=1)
    dices = {}
    for c in range(1, n_classes):
        pc, tc = (pred == c).float(), (targets == c).float()
        if tc.sum() == 0 and pc.sum() == 0:
            continue
        dices[c] = (2 * (pc * tc).sum() / (pc.sum() + tc.sum() + 1e-8)).item()
    return dices


@torch.no_grad()
def validate(model, val_dataset, device, dice_loss, n_labels, use_amp):
    model.eval()
    total_loss, all_dices = 0, []
    for i in range(len(val_dataset)):
        img_t, lab_t = val_dataset[i]
        img_t = img_t.unsqueeze(0).to(device)
        lab_t = lab_t.unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(img_t)
            loss = dice_loss(logits, lab_t)
        total_loss += loss.item()
        dices = compute_dice_per_class(logits, lab_t, n_labels)
        if dices:
            all_dices.append(np.mean(list(dices.values())))
    model.train()
    return total_loss / max(len(val_dataset), 1), np.mean(all_dices) if all_dices else 0.0


def save_checkpoint(model, optimizer, epoch, global_step, best_dice, path):
    torch.save({"epoch": epoch, "global_step": global_step, "best_dice": best_dice,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_e_roi198_augv2.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    torch.set_float32_matmul_precision("medium")
    device = torch.device(f"cuda:{args.gpu}")
    print(f"[Experiment E] GPU {args.gpu}: {torch.cuda.get_device_name(device)}")

    out_dir = Path(cfg["data"]["output_dir"])
    ckpt_dir, log_dir = out_dir / "checkpoints", out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    train_fine, train_coarse, val_coarse, val_images = discover_data(cfg)

    n_labels = cfg["volume"]["n_labels"]
    target_shape = tuple(cfg["volume"]["target_shape"])
    steps_per_epoch = train_cfg["steps_per_epoch"]
    use_amp = train_cfg["amp"]

    print("Loading label maps (fine + coarse pairs)...")
    fine_np = [nib.load(str(p)).get_fdata().astype(np.int32) for p in train_fine]
    coarse_np = [nib.load(str(p)).get_fdata().astype(np.int32) for p in train_coarse]

    gpu_synth = GPUSynthGeneratorFineV2(
        fine_labels_np=fine_np, coarse_labels_np=coarse_np,
        n_labels=n_labels, target_shape=target_shape,
        device=device, cfg=cfg.get("synth", {}),
    )
    del fine_np, coarse_np

    val_dataset = RealImageDataset(
        image_paths=val_images, label_paths=val_coarse, target_shape=target_shape,
    )

    model = build_model(cfg).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    criterion = DiceCELoss(n_labels).to(device)
    dice_loss = DiceLoss(n_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, global_step, best_dice = 0, 0, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", start_epoch * steps_per_epoch)
        best_dice = ckpt.get("best_dice", 0.0)
        print(f"  Resumed at epoch {start_epoch}, step {global_step}, dice {best_dice:.4f}")

    log_path = log_dir / "training_log.jsonl"
    print(f"\n{'='*60}")
    print(f"Experiment E: roi198 + full SynthSeg augmentation")
    print(f"  Augmentations: affine(+-15deg,0.8-1.2x) + elastic + LR-flip(50%)")
    print(f"  Background: 30%black/40%dim/30%standard")
    print(f"  Gamma: exp(N(0,0.5)), no additive noise, std range [0,30]")
    print(f"  Fine labels for intensity: 198, target: {n_labels} (roi22)")
    print(f"  Total steps: {train_cfg['epochs'] * steps_per_epoch}")
    print(f"{'='*60}\n")

    model.train()
    for epoch in range(start_epoch, train_cfg["epochs"]):
        t0 = time.time()
        epoch_losses = []
        pbar = tqdm(range(steps_per_epoch), desc=f"ExpE Epoch {epoch}", leave=False)
        for step in pbar:
            images, labels = gpu_synth.generate(batch_size=train_cfg["batch_size"])
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
        log_entry = {"epoch": epoch, "global_step": global_step,
                     "train_loss": float(avg_loss), "elapsed_s": elapsed,
                     "steps_per_sec": steps_per_epoch / elapsed}

        if (epoch + 1) % train_cfg["val_every"] == 0:
            val_loss, val_dice = validate(model, val_dataset, device, dice_loss, n_labels, use_amp)
            log_entry.update({"val_loss": float(val_loss), "val_dice": float(val_dice)})
            print(f"ExpE Epoch {epoch:4d} | step {global_step} | "
                  f"train={avg_loss:.4f} val={val_loss:.4f} dice={val_dice:.4f} | "
                  f"{elapsed:.1f}s ({steps_per_epoch / elapsed:.1f} steps/s)")
            if val_dice > best_dice:
                best_dice = val_dice
                save_checkpoint(model, optimizer, epoch, global_step, best_dice, ckpt_dir / "best.pt")
                print(f"  >> New best dice: {best_dice:.4f}")
        else:
            print(f"ExpE Epoch {epoch:4d} | step {global_step} | "
                  f"train={avg_loss:.4f} | {elapsed:.1f}s ({steps_per_epoch / elapsed:.1f} steps/s)")

        if (epoch + 1) % train_cfg["save_every"] == 0:
            save_checkpoint(model, optimizer, epoch, global_step, best_dice,
                            ckpt_dir / f"epoch_{epoch+1:04d}.pt")
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    save_checkpoint(model, optimizer, train_cfg["epochs"] - 1, global_step, best_dice,
                    ckpt_dir / "final.pt")
    print(f"\n[Experiment E] Done. Best val dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
