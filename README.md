# SPIMDash (SynthSegSPIM)

**SynthSeg-style deep learning segmentation for SPIM (lightsheet) mouse brain data.**

Replaces SPIMquant's registration-to-atlas pipeline (stages 1-6) with a single 3D U-Net that directly predicts atlas labels from downsampled SPIM images — no registration needed.

## Approach

We use the [SynthSeg](https://github.com/BBillot/SynthSeg) strategy: train on **purely synthetic images** generated on-the-fly from label maps. The network never sees a real image during training, learning to segment anatomy regardless of contrast, intensity, or noise. Our 52 existing SPIMquant outputs (atlas labels warped to subject space) provide the label maps.

**Key innovation:** All synthetic data generation runs **on GPU** (~30ms/sample) rather than CPU (22s/sample with scipy), eliminating the data pipeline bottleneck entirely.

## Setup

```bash
# Install dependencies via pixi
pixi install
```

## Training

```bash
# Launch training on GPU node (single A100, GPU-native synth generation)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pixi run python scripts/train.py --config configs/default.yaml --gpu 0

# Use --gpu 1 to run a parallel experiment on the second A100
# Resume from checkpoint:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pixi run python scripts/train.py --config configs/default.yaml --gpu 0 --resume outputs/checkpoints/best.pt
```

## Inference

```bash
# Single volume
pixi run python scripts/infer.py \
    --checkpoint outputs/checkpoints/best.pt \
    --config outputs/config.yaml \
    --input /path/to/SPIM.nii.gz \
    --output /path/to/dseg.nii.gz

# Batch mode
pixi run python scripts/infer.py \
    --checkpoint outputs/checkpoints/best.pt \
    --config outputs/config.yaml \
    --input-dir /path/to/level5_niis/ \
    --output-dir /path/to/predicted_dsegs/
```

## Evaluation

```bash
pixi run python scripts/evaluate.py \
    --pred-dir outputs/predictions/ \
    --gt-dir /nfs/khan/trainees/apooladi/brainhack/labels \
    --image-dir /nfs/khan/trainees/apooladi/brainhack/downsampled_lighsheet5 \
    --output-dir outputs/evaluation/
```

## Data

- **49 subjects** used (3 excluded for QC: 022, 032, 041), all Abeta channel level-5 downsampled
- **23 atlas labels** (ABAv3 roi-22), labels 1/2 (olfactory bulb) merged to background
- **Variable shapes** across subjects (148-272 x 312-392 x 109-226 voxels)
- **Anisotropic voxels**: 0.052 x 0.052 x 0.035 mm
- Target shape: **224 x 320 x 192** (center pad/crop, divisible by 32 for 5-level U-Net)
- Split: 39 train / 10 validation (seeded random)

## Model

Custom 3D U-Net matching the SynthSeg architecture (pure PyTorch):

- **5 encoder levels**, channels: [24, 48, 96, 192, 384] (13.2M parameters)
- Conv3d(3x3x3) -> ELU -> BatchNorm3d, 2 convolutions per block
- MaxPool3d(2) downsampling, nearest-neighbor upsampling
- Skip connections via concatenation
- Output: 1x1x1 conv -> raw logits (no softmax in model)
- No dropout, no residual connections

## Training Protocol

- **Loss:** Dice + CrossEntropy from step 0
- **Optimizer:** Adam, constant lr=1e-4, no weight decay, no scheduler
- **60 epochs x 500 steps/epoch** = 30,000 total steps
- **Batch size:** 2 (single A100 40GB, mixed precision)
- **Validation:** every 3 epochs on 10 real image+label pairs

## Synthetic Data Generation (GPU-native)

Each training step generates a fresh synthetic image entirely on GPU:

1. Pick random label map from GPU memory pool
2. Elastic deformation via `grid_sample` (nearest-neighbor to keep labels discrete)
3. Random intensity per region (vectorized LUT sampling)
4. Smooth multiplicative bias field (trilinear upsample of small random tensor)
5. Gaussian noise
6. Gaussian blur (separable 1D convolutions)
7. Gamma augmentation
8. Normalize to [0, 1]

## Project Structure

```
SPIMDash/
├── configs/
│   └── default.yaml          # all hyperparameters and paths
├── models/
│   ├── unet.py               # custom 5-level SynthSeg 3D U-Net
│   └── losses.py             # Dice, DiceCE, WeightedL2 losses
├── utils/
│   ├── gpu_synth.py          # GPU-native synthetic data generator
│   └── synth_generator.py    # CPU fallback generator + RealImageDataset
├── scripts/
│   ├── train.py              # plain PyTorch training loop (GPU-native synth)
│   ├── infer.py              # inference (single or batch)
│   ├── evaluate.py           # per-region Dice + QC overlays
│   ├── inspect_data.py       # data inspection utility
│   └── test_gpu_memory.py    # GPU memory verification
└── pixi.toml                 # dependency management
```

## Hardware

- **2x NVIDIA A100-PCIE-40GB** (SLURM node rri-cbs-v1)
- Training uses 1 GPU; second available for parallel experiments
- Mixed precision (AMP) + TF32 matmul enabled
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to prevent memory fragmentation
