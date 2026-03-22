#!/bin/bash
set -euo pipefail

# Stage 2: Qwen3-8B full fine-tuning on vast.ai (H100)
#
# eff_batch = 16 × 8 = 128, 3 epochs, packing (~3x throughput)
# ~10.5 hours on H100 80GB
#
# Usage:
#   nohup bash run.sh 2>&1 &
#   tail -f train.log

cd /workspace/stage2

# --- Common setup ---
source /workspace/setup.sh

# --- Check prerequisites ---
[ ! -f "/workspace/data/Pet_Supplies_conversations_train.parquet" ] && echo "ERROR: no train data in /workspace/data/" && exit 1
[ ! -d "/workspace/stage1/output/stage1_8b/final" ] && echo "ERROR: Stage 1 model not found. Run Stage 1 first." && exit 1
echo ">>> Data & Stage 1 model OK"

# --- Train ---
echo ">>> Starting Stage 2..."
python3 train_8b.py \
    --stage1-model /workspace/stage1/output/stage1_8b/final \
    --train-file /workspace/data/Pet_Supplies_conversations_train.parquet \
    --val-file /workspace/data/Pet_Supplies_conversations_val.parquet \
    --output-dir output \
    --max-seq-length 512 \
    --lr 2e-5 \
    --batch-size 16 \
    --grad-accum 8 \
    --epochs 3 \
    --warmup-ratio 0.03 \
    --weight-decay 0.01 \
    --packing \
    --snapshot-steps 2000 \
    --max-snapshots 3 \
    --eval-steps 500 \
    --sid-eval-samples 200 \
    --logging-steps 25 \
    --no-wandb \
    --no-torch-compile \
    "$@" \
    2>&1 | tee train.log

echo ">>> Done! Model at output/final/"
