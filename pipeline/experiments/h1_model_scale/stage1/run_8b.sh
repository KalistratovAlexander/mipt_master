#!/bin/bash
set -euo pipefail

# Stage 1: Qwen3-8B vocab expansion
# Adds 1027 SID tokens, trains only embeddings (~0.3% of params)
# ~30-40 min on H100 NVL

cd /workspace/stage1

source /workspace/setup.sh

[ ! -f "/workspace/data/Pet_Supplies_conversations_train.parquet" ] && echo "ERROR: no train data" && exit 1
echo ">>> Data OK"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo ">>> Starting Stage 1 (8B)..."
python3 train_1.8b.py \
    --model-name Qwen/Qwen3-8B \
    --train-file /workspace/data/Pet_Supplies_conversations_train.parquet \
    --val-file /workspace/data/Pet_Supplies_conversations_val.parquet \
    --output-dir output/stage1_8b \
    --max-seq-length 320 \
    --max-train-samples 64000 \
    --max-val-samples 2000 \
    --lr 1e-3 \
    --batch-size 32 \
    --grad-accum 4 \
    --max-steps 2000 \
    --warmup-steps 100 \
    --logging-steps 50 \
    --eval-steps 500 \
    --save-steps 500 \
    --no-torch-compile \
    --no-wandb \
    "$@" \
    2>&1 | tee train.log

echo ">>> Stage 1 done! Model at output/stage1_8b/final/"
