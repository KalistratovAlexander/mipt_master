#!/bin/bash
set -euo pipefail

# Stage 1: Qwen3-1.7B vocab expansion
# Adds 1027 SID tokens, trains only embeddings (~0.3% of params)
#
# Usage:
#   nohup bash run.sh 2>&1 &
#   tail -f train.log

cd /workspace/stage1

# --- Common setup ---
source /workspace/setup.sh

# --- Check data ---
[ ! -f "/workspace/data/Pet_Supplies_conversations_train.parquet" ] && echo "ERROR: no train data in /workspace/data/" && exit 1
echo ">>> Data OK"

# --- Train ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo ">>> Starting Stage 1..."
python3 train_1.8b.py \
    --model-name Qwen/Qwen3-1.7B \
    --train-file /workspace/data/Pet_Supplies_conversations_train.parquet \
    --val-file /workspace/data/Pet_Supplies_conversations_val.parquet \
    --output-dir output/stage1_1.8b \
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

echo ">>> Stage 1 done! Model at output/stage1_1.8b/final/"
