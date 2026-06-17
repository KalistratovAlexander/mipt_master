#!/bin/bash
set -euo pipefail

# Stage 2: Qwen3-0.6B full fine-tuning
# eff_batch = 128, 1 epoch, packing — ~1.5 hr on H100

cd /workspace/stage2

SMOKE_TEST="${SMOKE_TEST:-0}"
if [ "$SMOKE_TEST" = "0" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set — push will fail after training. Run: export HF_TOKEN=hf_..."
    exit 1
fi

source /workspace/setup.sh

[ ! -f "/workspace/data/Pet_Supplies_conversations_train.parquet" ] && echo "ERROR: no train data" && exit 1
[ ! -d "/workspace/stage1/output/stage1_0.6b/final" ] && echo "ERROR: Stage 1 model not found" && exit 1
echo ">>> Data & Stage 1 model OK"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo ">>> Starting Stage 2 (0.6B)..."
python3 train_1.8b.py \
    --stage1-model /workspace/stage1/output/stage1_0.6b/final \
    --train-file /workspace/data/Pet_Supplies_conversations_train.parquet \
    --val-file /workspace/data/Pet_Supplies_conversations_val.parquet \
    --output-dir output \
    --max-seq-length 320 \
    --lr 2e-5 \
    --batch-size 128 \
    --grad-accum 1 \
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
    --max-train-samples 1180000 \
    "$@" \
    2>&1 | tee train.log

echo ">>> Done! Model at output/final/"

if [ "$SMOKE_TEST" = "0" ]; then
    HF_REPO="${HF_REPO:-kalistratov/qwen3-0.6b-sid-pet-1ep-seed42}"
    echo ">>> Pushing to HF: $HF_REPO"
    [ ! -f output/final/chat_template.jinja ] && cp /workspace/stage1/output/stage1_0.6b/final/chat_template.jinja output/final/ 2>/dev/null || true
    hf upload "$HF_REPO" output/final --repo-type=model --private \
        && echo ">>> HF push OK" \
        || echo ">>> WARNING: HF push failed"
else
    echo ">>> SMOKE_TEST=1 — skipping HF push"
fi
