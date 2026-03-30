#!/usr/bin/env python3
"""Optuna sweep for synthetic data generation hyperparameters.

Auto-detects GPUs and runs roi198 on GPU 0, roi-all on GPU 1 in parallel.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pixi run python scripts/sweep_synth_params.py
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pixi run python scripts/sweep_synth_params.py --n-trials 200
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "/nfs/khan/trainees/apooladi/brainhack"
TARGET_SHAPE = (224, 320, 192)
N_LABELS = 23


def run_sweep(gpu_id, fine_label_name, fine_dir, n_trials, n_synth_per_trial):
    """Run one sweep on one GPU. Called as a subprocess."""
    import torch
    import numpy as np
    import optuna
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.gpu_synth_sweep import GPUSynthSweep, conform_and_to_tensor
    from utils.sweep_features import (
        compute_features_gpu, load_real_features, compute_real_stats,
        compute_real_nn_stats,
    )
    from utils.discriminability import compute_discriminability

    device = torch.device(f"cuda:{gpu_id}")
    study_name = f"sweep_{fine_label_name}"
    coarse_dir = f"{BASE}/labels"
    out_dir = Path("outputs/sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / f"{study_name}.db"

    print(f"\n[GPU {gpu_id}] {fine_label_name}: {torch.cuda.get_device_name(device)}")

    # ── Pre-compute real features (cached) ──
    real_feats, real_stains = load_real_features(device)
    real_mean, real_std = compute_real_stats(real_feats)
    real_median_nn, real_p95_nn = compute_real_nn_stats(real_feats, real_mean, real_std)
    real_feats_std = (real_feats - real_mean) / real_std
    real_spread = float(np.trace(np.cov(real_feats_std.T)))
    print(f"[GPU {gpu_id}] Real: {len(real_feats)} images, median NN: {real_median_nn:.3f}")

    # ── Discriminability threshold (cached) ──
    disc_cache = out_dir / f"disc_threshold_gpu{gpu_id}.npy"
    if disc_cache.exists():
        disc_threshold = float(np.load(disc_cache))
    else:
        print(f"[GPU {gpu_id}] Computing disc threshold from Exp A...")
        from utils.gpu_synth import GPUSynthGenerator
        exclude = {"022", "032", "041"}
        paths = sorted(Path(coarse_dir).glob("*.nii*"))
        paths = [p for p in paths if p.name.split("_")[0] not in exclude]
        labels_np = [nib.load(str(p)).get_fdata().astype(np.int32) for p in paths]
        gen = GPUSynthGenerator(labels_np, N_LABELS, TARGET_SHAPE, device,
                                {"intensity_mu_range": [0, 300], "intensity_sigma_range": [5, 30],
                                 "deform_scale": 3.0, "deform_grid_spacing": 32,
                                 "bias_field_coeffs": 4, "bias_field_strength": 0.4,
                                 "blur_sigma_range": [0, 1.5], "noise_std_range": [0, 25]})
        del labels_np
        scores = []
        for _ in range(20):
            imgs, labs = gen.generate(batch_size=1)
            scores.append(compute_discriminability(imgs, labs))
        disc_threshold = float(np.percentile(scores, 25))
        np.save(disc_cache, disc_threshold)
        del gen
        torch.cuda.empty_cache()
    print(f"[GPU {gpu_id}] Disc threshold: {disc_threshold:.4f}")

    # ── Load label maps + pre-compute n_fine_labels ──
    print(f"[GPU {gpu_id}] Loading label maps...")
    exclude = {"022", "032", "041"}
    coarse_paths = sorted(Path(coarse_dir).glob("*.nii*"))
    fine_paths = sorted(Path(fine_dir).glob("*.nii*"))
    coarse_dict = {p.name.split("_")[0]: p for p in coarse_paths}
    fine_dict = {p.name.split("_")[0]: p for p in fine_paths}
    common = sorted(set(coarse_dict) & set(fine_dict) - exclude)

    fine_np_list = [nib.load(str(fine_dict[s])).get_fdata().astype(np.int32) for s in common]
    coarse_np_list = [nib.load(str(coarse_dict[s])).get_fdata().astype(np.int32) for s in common]

    # Pre-compute n_fine_labels ONCE
    all_fine = set()
    for f in fine_np_list:
        all_fine.update(np.unique(f).tolist())
    n_fine_labels = max(all_fine) + 1
    print(f"[GPU {gpu_id}] {len(common)} subjects, {len(all_fine)} fine labels (max idx {n_fine_labels-1})")

    # Pre-load to GPU ONCE (reused across all trials via _preloaded=True)
    olf_labels = [1, 2]
    fine_tensors = []
    coarse_tensors = []
    for fnp, cnp in zip(fine_np_list, coarse_np_list):
        fine_tensors.append(conform_and_to_tensor(fnp, TARGET_SHAPE).to(device))
        cm = cnp.copy()
        for ol in olf_labels:
            cm[cm == ol] = 0
        coarse_tensors.append(conform_and_to_tensor(cm, TARGET_SHAPE).to(device))
    del fine_np_list, coarse_np_list

    batch_size = 2
    n_batches = n_synth_per_trial // batch_size

    def objective(trial):
        cfg = {
            "activation_prob_lo": trial.suggest_float("activation_prob_lo", 0.15, 0.5),
            "activation_prob_hi": trial.suggest_float("activation_prob_hi", 0.5, 1.0),
            "inactive_intensity_max": trial.suggest_float("inactive_intensity_max", 5, 50),
            "inactive_std_max": trial.suggest_float("inactive_std_max", 1, 15),
            "boundary_blur_sigma_max": trial.suggest_float("boundary_blur_sigma_max", 0.5, 3.0),
            "boundary_blur_prob": trial.suggest_float("boundary_blur_prob", 0.3, 1.0),
            "texture_noise_amplitude": trial.suggest_float("texture_noise_amplitude", 0.05, 0.5),
            "texture_base_resolution": trial.suggest_int("texture_base_resolution", 8, 32),
            "texture_prob": trial.suggest_float("texture_prob", 0.3, 1.0),
            "intensity_mean_hi": trial.suggest_float("intensity_mean_hi", 100, 500),
            "intensity_std_lo": trial.suggest_float("intensity_std_lo", 0, 10),
            "intensity_std_hi": trial.suggest_float("intensity_std_hi", 15, 50),
            "gamma_std": trial.suggest_float("gamma_std", 0.3, 1.5),
            "bias_field_std": trial.suggest_float("bias_field_std", 0.3, 1.5),
            "deform_scale": 3.0,
            "deform_grid_spacing": 32,
            "bias_field_coeffs": 4,
        }

        # Reuse pre-loaded GPU tensors — no numpy scanning, no CPU->GPU transfer
        gen = GPUSynthSweep(
            fine_tensors, coarse_tensors, N_LABELS, TARGET_SHAPE, device,
            cfg=cfg, n_fine_labels=n_fine_labels, _preloaded=True,
        )

        all_feats = []
        disc_scores = []
        for bi in range(n_batches):
            imgs, labs = gen.generate(batch_size=batch_size)
            if bi < 5:
                disc_scores.append(compute_discriminability(imgs, labs))
            for b in range(imgs.shape[0]):
                mask_t = (labs[b] > 0).float()
                feat = compute_features_gpu(imgs[b, 0], mask_t, device)
                all_feats.append(feat)

        disc = float(np.mean(disc_scores)) if disc_scores else 0.0
        if disc < disc_threshold:
            trial.set_user_attr("discriminability", disc)
            return -1.0

        synth_feats = np.stack(all_feats)
        valid = ~np.any(np.isnan(synth_feats), axis=1)
        synth_feats = synth_feats[valid]
        if len(synth_feats) < 10:
            return -1.0

        synth_feats_std = (synth_feats - real_mean) / real_std

        from scipy.spatial.distance import cdist
        dists = cdist(real_feats_std, synth_feats_std)
        nn_dists = dists.min(axis=1)
        sorted_d = np.sort(nn_dists)
        fracs = np.linspace(0, 1, len(sorted_d))
        mask = sorted_d <= real_p95_nn
        if mask.sum() < 2:
            auc = 0.0
        else:
            auc = float(np.trapezoid(fracs[mask], sorted_d[mask]) / real_p95_nn)

        try:
            synth_spread = float(np.trace(np.cov(synth_feats_std.T)))
        except Exception:
            synth_spread = 0.0
        spread_ratio = min(synth_spread / (real_spread + 1e-8), 5.0) / 5.0

        obj = 0.7 * auc + 0.3 * spread_ratio

        trial.set_user_attr("coverage_auc", float(auc))
        trial.set_user_attr("spread_ratio", float(spread_ratio))
        trial.set_user_attr("discriminability", float(disc))
        trial.set_user_attr("mean_nn_dist", float(nn_dists.mean()))
        trial.set_user_attr("pct_covered_at_median", float((nn_dists <= real_median_nn).mean()))

        return obj

    # ── Run Optuna ──
    print(f"\n[GPU {gpu_id}] Starting {study_name}: {n_trials} trials")
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42 + gpu_id),
    )

    def progress(study, trial):
        if (trial.number + 1) % 10 == 0:
            best = study.best_trial
            print(f"[GPU {gpu_id}] Trial {trial.number+1}/{n_trials} | "
                  f"best={best.value:.4f} auc={best.user_attrs.get('coverage_auc',0):.4f} "
                  f"covered={best.user_attrs.get('pct_covered_at_median',0):.1%}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, callbacks=[progress])

    print(f"\n[GPU {gpu_id}] {study_name} DONE. Best: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--n-synth-per-trial", type=int, default=300)
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPUs")

    sweeps = [
        (0, "roi198", f"{BASE}/roi_labels_198"),
        (1, "roiall", f"{BASE}/roi_labels_all"),
    ]
    sweeps = sweeps[:n_gpus]  # only run what we have GPUs for

    if len(sweeps) == 1:
        # Single GPU: run sequentially
        gpu_id, name, fine_dir = sweeps[0]
        run_sweep(gpu_id, name, fine_dir, args.n_trials, args.n_synth_per_trial)
    else:
        # Multi-GPU: run in parallel
        procs = []
        for gpu_id, name, fine_dir in sweeps:
            p = mp.Process(target=run_sweep,
                           args=(gpu_id, name, fine_dir, args.n_trials, args.n_synth_per_trial))
            p.start()
            procs.append((p, name))
            print(f"Started {name} sweep on GPU {gpu_id} (pid {p.pid})")

        for p, name in procs:
            p.join()
            print(f"{name} sweep finished (exit code {p.exitcode})")

    print("\nAll sweeps complete. Run sweep_analysis.py to analyze results.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
