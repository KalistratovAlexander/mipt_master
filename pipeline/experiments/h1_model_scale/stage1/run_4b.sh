#!/bin/bash
set -euo pipefail

# Stage 1: Qwen3-4B vocab expansion
# ~25 min on H100, peak VRAM ~20 GB
#
# Usage: bash run.sh   (called from pack.sh deployment as stage1/run.sh)

cd /workspace/stage1

source /workspace/setup.sh

[ ! -f "/workspace/data/Pet_Supplies_conversations_train.parquet" ] && echo "ERROR: no train data" && exit 1
echo ">>> Data OK"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo ">>> Starting Stage 1 (4B)..."
python3 train_1.8b.py \
    --model-name Qwen/Qwen3-4B \
    --train-file /workspace/data/Pet_Supplies_conversations_train.parquet \
    --val-file /workspace/data/Pet_Supplies_conversations_val.parquet \
    --output-dir output/stage1_4b \
    --max-seq-length 320 \
    --max-train-samples 64000 \
    --max-val-samples 2000 \
    --lr 1e-3 \
    --batch-size 64 \
    --grad-accum 2 \
    --max-steps 2000 \
    --warmup-steps 100 \
    --logging-steps 50 \
    --eval-steps 500 \
    --save-steps 500 \
    --no-torch-compile \
    --no-wandb \
    "$@" \
    2>&1 | tee train.log

echo ">>> Stage 1 done! Model at output/stage1_4b/final/"
