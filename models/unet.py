"""
Custom 3D U-Net matching SynthSeg's architecture.

5 encoder levels, Conv→ELU→BatchNorm, no dropout, no residual connections.
forward() returns raw logits (no softmax).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Single convolution: Conv3d(3x3x3) -> ELU -> BatchNorm3d."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ELU(inplace=True)
        self.bn = nn.BatchNorm3d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.act(self.conv(x)))


class EncoderBlock(nn.Module):
    """Two ConvBlocks then MaxPool3d(2).

    Returns (skip_features, pooled_output).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        return x, self.pool(x)


class DecoderBlock(nn.Module):
    """Upsample(nearest, 2x) + concat skip + two ConvBlocks."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBlock(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Handle size mismatches from odd dimensions after pooling
        if x.shape != skip.shape:
            x = F.pad(x, [
                0, skip.shape[4] - x.shape[4],
                0, skip.shape[3] - x.shape[3],
                0, skip.shape[2] - x.shape[2],
            ])
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SynthSegUNet(nn.Module):
    """5-level 3D U-Net matching SynthSeg architecture.

    forward() returns RAW LOGITS (no softmax).

    Encoder channel flow:
        Level 1: 1 -> 24 -> 24, then pool
        Level 2: 24 -> 48 -> 48, then pool
        Level 3: 48 -> 96 -> 96, then pool
        Level 4: 96 -> 192 -> 192, then pool
        Bottleneck: 192 -> 384 -> 384 (no pool)

    Decoder channel flow (upsample + concat skip + 2x conv):
        Level 4: upsample 384, cat skip 192 -> 576 in, conv to 192
        Level 3: upsample 192, cat skip 96  -> 288 in, conv to 96
        Level 2: upsample 96,  cat skip 48  -> 144 in, conv to 48
        Level 1: upsample 48,  cat skip 24  ->  72 in, conv to 24

    Output: Conv3d(24, n_labels, kernel_size=1) -- logits, NO softmax
    """

    def __init__(self, in_channels: int = 1, n_labels: int = 23, base_features: int = 24):
        super().__init__()
        f = base_features  # 24
        feats = [f, f * 2, f * 4, f * 8, f * 16]  # [24, 48, 96, 192, 384]

        # Encoder
        self.enc1 = EncoderBlock(in_channels, feats[0])  # 1 -> 24
        self.enc2 = EncoderBlock(feats[0], feats[1])      # 24 -> 48
        self.enc3 = EncoderBlock(feats[1], feats[2])      # 48 -> 96
        self.enc4 = EncoderBlock(feats[2], feats[3])      # 96 -> 192

        # Bottleneck (no pool)
        self.bottleneck1 = ConvBlock(feats[3], feats[4])  # 192 -> 384
        self.bottleneck2 = ConvBlock(feats[4], feats[4])  # 384 -> 384

        # Decoder
        self.dec4 = DecoderBlock(feats[4], feats[3], feats[3])  # 384+192=576 -> 192
        self.dec3 = DecoderBlock(feats[3], feats[2], feats[2])  # 192+96=288 -> 96
        self.dec2 = DecoderBlock(feats[2], feats[1], feats[1])  # 96+48=144 -> 48
        self.dec1 = DecoderBlock(feats[1], feats[0], feats[0])  # 48+24=72 -> 24

        # Output head: 1x1x1 conv, NO activation
        self.out_conv = nn.Conv3d(feats[0], n_labels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip1, x = self.enc1(x)   # skip1: 24-ch at full res
        skip2, x = self.enc2(x)   # skip2: 48-ch at 1/2
        skip3, x = self.enc3(x)   # skip3: 96-ch at 1/4
        skip4, x = self.enc4(x)   # skip4: 192-ch at 1/8

        # Bottleneck
        x = self.bottleneck1(x)    # 384-ch at 1/16
        x = self.bottleneck2(x)

        # Decoder
        x = self.dec4(x, skip4)    # 192-ch at 1/8
        x = self.dec3(x, skip3)    # 96-ch at 1/4
        x = self.dec2(x, skip2)    # 48-ch at 1/2
        x = self.dec1(x, skip1)    # 24-ch at full res

        return self.out_conv(x)    # n_labels-ch logits


def build_model(cfg: dict) -> nn.Module:
    """Build segmentation model from config."""
    model_cfg = cfg["model"]
    n_labels = cfg["volume"]["n_labels"]

    return SynthSegUNet(
        in_channels=model_cfg.get("in_channels", 1),
        n_labels=n_labels,
        base_features=model_cfg.get("base_features", 24),
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
