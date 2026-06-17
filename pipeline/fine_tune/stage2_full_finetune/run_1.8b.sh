#!/bin/bash
set -euo pipefail

# Stage 2: Qwen3-1.7B full fine-tuning on vast.ai
#
# eff_batch = 64 × 2 = 128, 1 epoch, packing (~3x throughput)
#
# Usage:
#   nohup bash run.sh 2>&1 &
#   tail -f train.log

cd /workspace/stage2

SMOKE_TEST="${SMOKE_TEST:-0}"
if [ "$SMOKE_TEST" = "0" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set — push will fail after training. Run: export HF_TOKEN=hf_..."
    exit 1
fi

# --- Common setup ---
source /workspace/setup.sh

# --- Check prerequisites ---
[ ! -f "/workspace/data/Pet_Supplies_conversations_train.parquet" ] && echo "ERROR: no train data in /workspace/data/" && exit 1
[ ! -d "/workspace/stage1/output/stage1_1.8b/final" ] && echo "ERROR: Stage 1 model not found. Run Stage 1 first." && exit 1
echo ">>> Data & Stage 1 model OK"

# --- Train ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo ">>> Starting Stage 2..."
python3 train_1.8b.py \
    --stage1-model /workspace/stage1/output/stage1_1.8b/final \
    --train-file /workspace/data/Pet_Supplies_conversations_train.parquet \
    --val-file /workspace/data/Pet_Supplies_conversations_val.parquet \
    --output-dir output \
    --max-seq-length 320 \
    --lr 2e-5 \
    --batch-size 64 \
    --grad-accum 2 \
    --epochs 1 \
    --warmup-ratio 0.03 \
    --weight-decay 0.01 \
    --packing \
    --snapshot-steps 2000 \
    --max-snapshots 3 \
    --eval-steps 500 \
    --sid-eval-samples 200 \
    --logging-steps 25 \
    --no-torch-compile \
    --no-wandb \
    "$@" \
    2>&1 | tee train.log

echo ">>> Done! Model at output/final/"

if [ "$SMOKE_TEST" = "0" ]; then
    HF_REPO="${HF_REPO:-kalistratov/qwen3-1.7b-sid-pet-1ep-seed42}"
    echo ">>> Pushing to HF: $HF_REPO"
    hf upload "$HF_REPO" output/final --repo-type=model --private \
        && echo ">>> HF push OK" \
        || echo ">>> WARNING: HF push failed; model still in output/final/"
else
    echo ">>> SMOKE_TEST=1 — skipping HF push"
fi
