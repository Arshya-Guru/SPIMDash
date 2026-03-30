"""Structural discriminability metric for synthetic images.

Measures how visually distinguishable labeled regions are from each other.
Used as a guardrail in the hyperparameter sweep to prevent degenerate solutions.
"""

import torch
import numpy as np


@torch.no_grad()
def compute_discriminability(images: torch.Tensor, label_maps: torch.Tensor) -> float:
    """Compute mean discriminability across a batch of image+label pairs.

    For each image:
    1. For each label (excl background), compute mean intensity
    2. Sort the means
    3. Compute gaps between adjacent sorted means
    4. Discriminability = mean(gaps) / median(per-label stds)

    Args:
        images: (B, 1, D, H, W) float tensor
        label_maps: (B, D, H, W) long tensor

    Returns:
        Mean discriminability across the batch.
    """
    B = images.shape[0]
    scores = []

    for b in range(B):
        img = images[b, 0]  # (D, H, W)
        lab = label_maps[b]  # (D, H, W)

        unique_labels = torch.unique(lab)
        unique_labels = unique_labels[unique_labels > 0]  # exclude background

        if len(unique_labels) < 2:
            continue

        means = []
        stds = []
        for l in unique_labels:
            mask = lab == l
            vals = img[mask]
            if vals.numel() < 10:
                continue
            means.append(vals.mean().item())
            stds.append(vals.std().item())

        if len(means) < 2:
            continue

        sorted_means = sorted(means)
        gaps = [sorted_means[i+1] - sorted_means[i] for i in range(len(sorted_means)-1)]
        mean_gap = np.mean(gaps)
        noise_level = np.median(stds) + 1e-8

        scores.append(mean_gap / noise_level)

    return float(np.mean(scores)) if scores else 0.0
