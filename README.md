# SPIMDash (SynthSegSPIM)

**SynthSeg-style deep learning segmentation for SPIM (lightsheet) brain data.**

Replaces SPIMquant's registration-to-atlas pipeline (stages 1-6) with a single 3D U-Net that directly predicts atlas labels from downsampled SPIM images.

## Setup

```bash
# Install dependencies via pixi
pixi install

# Inspect data (already done - 52 subjects, 23 labels, variable shapes)
pixi run python scripts/inspect_data.py \
    --label-dir /nfs/khan/trainees/apooladi/brainhack/labels \
    --image-dir /nfs/khan/trainees/apooladi/brainhack/downsampled_lighsheet5
```

## Usage

### 1. Test GPU memory (run on GPU node)

```bash
pixi run python scripts/test_gpu_memory.py --config configs/default.yaml
```

### 2. Train

```bash
pixi run python scripts/train.py --config configs/default.yaml
```

### 3. Inference

```bash
pixi run python scripts/infer.py \
    --checkpoint outputs/checkpoints/best.pt \
    --config outputs/config.yaml \
    --input /path/to/SPIM.nii.gz \
    --output /path/to/dseg.nii.gz
```

### 4. Evaluate

```bash
pixi run python scripts/evaluate.py \
    --pred-dir outputs/predictions/ \
    --gt-dir /nfs/khan/trainees/apooladi/brainhack/labels \
    --image-dir /nfs/khan/trainees/apooladi/brainhack/downsampled_lighsheet5 \
    --output-dir outputs/evaluation/
```

## Data Summary

- **52 subjects**, all Abeta channel level-5 downsampled
- **23 atlas labels** (0-22, where 0=background)
- **Variable shapes** across subjects (148-272 x 312-392 x 109-226)
- **Anisotropic voxels**: 0.052 x 0.052 x 0.035 mm
- Target shape for training: **224 x 320 x 192** (center pad/crop)

## Project Structure

```
SPIMDash/
├── configs/
│   └── default.yaml          # all hyperparameters and paths
├── models/
│   ├── unet.py               # MONAI 3D U-Net wrapper
│   └── losses.py             # Dice + CE loss
├── utils/
│   └── synth_generator.py    # SynthSeg-style on-the-fly data generation
├── scripts/
│   ├── inspect_data.py       # data inspection
│   ├── test_gpu_memory.py    # GPU memory verification
│   ├── train.py              # main training loop
│   ├── infer.py              # inference on new subjects
│   └── evaluate.py           # Dice evaluation + QC images
└── pixi.toml                 # dependency management
```
