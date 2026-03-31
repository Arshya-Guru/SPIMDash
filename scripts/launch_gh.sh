#!/bin/bash
# Launch Experiments G and H on separate GPUs
# Safe to run repeatedly — auto-resume handles continuation

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."

mkdir -p outputs/experiment_g/logs outputs/experiment_g/checkpoints
mkdir -p outputs/experiment_h/logs outputs/experiment_h/checkpoints

echo "=== Launching Experiment G (roi198, sweep Trial 257) on GPU 0 ==="
nohup pixi run python scripts/train_sweep.py \
    --config configs/experiment_g.yaml \
    --gpu 0 \
    > outputs/experiment_g/train.log 2>&1 &
echo "PID: $!"

echo "=== Launching Experiment H (roi-all, sweep Trial 257) on GPU 1 ==="
nohup pixi run python scripts/train_sweep.py \
    --config configs/experiment_h.yaml \
    --gpu 1 \
    > outputs/experiment_h/train.log 2>&1 &
echo "PID: $!"

echo ""
echo "Monitor:"
echo "  tail -f outputs/experiment_g/train.log"
echo "  tail -f outputs/experiment_h/train.log"
echo ""
echo "Check Dice:"
echo "  grep '\"type\": \"val\"' outputs/experiment_g/logs/training_log.jsonl | tail -5"
echo "  grep '\"type\": \"val\"' outputs/experiment_h/logs/training_log.jsonl | tail -5"
