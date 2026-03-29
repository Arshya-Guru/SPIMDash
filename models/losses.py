"""Loss functions for SynthSeg-style atlas segmentation.

Two-phase training:
  Phase 1 (warmup): WeightedL2Loss — MSE between logits and one-hot targets
  Phase 2 (main):   DiceLoss — soft Dice averaged over foreground classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedL2Loss(nn.Module):
    """MSE between raw logits and one-hot targets. For phase 1 warmup."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W) raw model output
            targets: (B, D, H, W) integer labels
        """
        targets_oh = F.one_hot(targets, self.n_classes)  # (B, D, H, W, C)
        targets_oh = targets_oh.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        return F.mse_loss(logits, targets_oh)


class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation.

    Applies softmax to logits internally, computes per-class Dice,
    averages over foreground classes (skipping background).
    """

    def __init__(self, n_classes: int, ignore_background: bool = True, smooth: float = 1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_background = ignore_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W) raw predictions
            targets: (B, D, H, W) integer labels
        """
        probs = F.softmax(logits, dim=1)

        targets_oh = F.one_hot(targets, self.n_classes)  # (B, D, H, W, C)
        targets_oh = targets_oh.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)

        start_ch = 1 if self.ignore_background else 0

        dice_sum = 0.0
        count = 0
        for c in range(start_ch, self.n_classes):
            pred_c = probs[:, c]
            true_c = targets_oh[:, c]

            intersection = (pred_c * true_c).sum()
            union = pred_c.sum() + true_c.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_sum += dice
            count += 1

        return 1.0 - dice_sum / max(count, 1)


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss. Kept as fallback option."""

    def __init__(self, n_classes: int, dice_weight: float = 1.0, ce_weight: float = 1.0):
        super().__init__()
        self.dice = DiceLoss(n_classes=n_classes, ignore_background=True)
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + \
               self.ce_weight * self.ce(logits, targets)


def build_loss(cfg: dict) -> nn.Module:
    """Build loss function from config. Default returns DiceLoss for testing/eval."""
    n_labels = cfg["volume"]["n_labels"]
    return DiceLoss(n_classes=n_labels)
