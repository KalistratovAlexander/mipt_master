#!/usr/bin/env bash
# H1 evaluation runner — single H100 80GB instance (vast.ai).
# Compares 4 model sizes: 0.6B, 1.7B, 4B, 8B (all fine-tuned with SID tokens).
# Estimated wall-clock: ~12h. Cost ~$16 at $1.34/h.
#
# Layout assumed (all paths relative to invocation dir):
#   pipeline/evaluation/{evaluate_unified.py,stat_tests.py}
#   data/embeds/Pet_Supplies_items_with_semantic_ids.parquet  (+ other parquet files)
#   data/sequences/Pet_Supplies_sequences_with_sid_train.parquet
#
# Usage:
#   bash pipeline/evaluation/run_h1.sh [output_dir]
#
# Each python call has --resume, so an OOM/restart picks up where it left off.

set -euo pipefail

# Auto-install dependencies if running on vast.ai
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f /tmp/.vast_eval_setup_done ]; then
    bash "$SCRIPT_DIR/setup_eval.sh"
fi

OUT_DIR="${1:-h1_artifacts/evaluation/canon_run}"
mkdir -p "$OUT_DIR"

EVAL=pipeline/evaluation/evaluate_unified.py
STAT=pipeline/evaluation/stat_tests.py
DATA_DIR=data                # evaluator auto-finds embeds/ + semantic_llm_training/
CATALOG=data/embeds/Pet_Supplies_items_with_semantic_ids.parquet
TRAIN_SEQ=data/sequences/Pet_Supplies_sequences_with_sid_train.parquet

MODEL_0P6B=kalistratov/qwen3-0.6b-sid-pet-1ep-seed42
MODEL_1P7B=kalistratov/qwen3-1.7b-sid-pet-1ep-seed42
MODEL_4B=kalistratov/qwen3-4b-sid-pet-1ep-seed42
MODEL_8B=kalistratov/qwen3-8b-sid-pet-1ep-seed42
BASE_0P6B=Qwen/Qwen3-0.6B
BASE_1P7B=Qwen/Qwen3-1.7B
BASE_4B=Qwen/Qwen3-4B
BASE_8B=Qwen/Qwen3-8B

COMMON=(
  --data-dir "$DATA_DIR"
  --samples-per-task 1500
  --samples-per-task-text 500
  --seed 42
  --beam-size 10
  --attn-impl flash_attention_2
  --cosine-model Qwen/Qwen3-Embedding-0.6B
  --bench-batch-size 1,32
)

PPL_ONLY=(
  --data-dir "$DATA_DIR"
  --samples-per-task 0
  --skip-benchmark
  --skip-cosine-sim
)

step() {
  local label=$1; shift
  echo
  echo "=========================================="
  echo "[$label] $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=========================================="
  python3 "$@"
}

step "1/11 fine-tuned 0.6B" "$EVAL" \
  --model-path "$MODEL_0P6B" \
  --model-name "0.6B" \
  --output "$OUT_DIR/eval_0p6b.json" \
  --resume \
  "${COMMON[@]}"

step "2/11 fine-tuned 1.7B" "$EVAL" \
  --model-path "$MODEL_1P7B" \
  --model-name "1.7B" \
  --output "$OUT_DIR/eval_1p7b.json" \
  --resume \
  "${COMMON[@]}"

step "3/11 fine-tuned 4B" "$EVAL" \
  --model-path "$MODEL_4B" \
  --model-name "4B" \
  --output "$OUT_DIR/eval_4b.json" \
  --resume \
  "${COMMON[@]}"

step "4/11 fine-tuned 8B" "$EVAL" \
  --model-path "$MODEL_8B" \
  --model-name "8B" \
  --output "$OUT_DIR/eval_8b.json" \
  --resume \
  "${COMMON[@]}"

step "5/11 base PPL 0.6B" "$EVAL" \
  --model-path "$BASE_0P6B" \
  --model-name "0.6B-base" \
  --output "$OUT_DIR/ppl_0p6b_base.json" \
  --resume \
  "${PPL_ONLY[@]}"

step "6/11 base PPL 1.7B" "$EVAL" \
  --model-path "$BASE_1P7B" \
  --model-name "1.7B-base" \
  --output "$OUT_DIR/ppl_1p7b_base.json" \
  --resume \
  "${PPL_ONLY[@]}"

step "7/11 base PPL 4B" "$EVAL" \
  --model-path "$BASE_4B" \
  --model-name "4B-base" \
  --output "$OUT_DIR/ppl_4b_base.json" \
  --resume \
  "${PPL_ONLY[@]}"

step "8/11 base PPL 8B" "$EVAL" \
  --model-path "$BASE_8B" \
  --model-name "8B-base" \
  --output "$OUT_DIR/ppl_8b_base.json" \
  --resume \
  "${PPL_ONLY[@]}"

step "9/11 stat_tests 0.6B vs 1.7B" "$STAT" \
  --eval-1p7b "$OUT_DIR/eval_0p6b.json" \
  --eval-8b   "$OUT_DIR/eval_1p7b.json" \
  --catalog "$CATALOG" \
  --train-sequences "$TRAIN_SEQ" \
  --output "$OUT_DIR/h1_stat_tests_0p6b_vs_1p7b.json"

step "10/11 stat_tests 1.7B vs 4B" "$STAT" \
  --eval-1p7b "$OUT_DIR/eval_1p7b.json" \
  --eval-8b   "$OUT_DIR/eval_4b.json" \
  --catalog "$CATALOG" \
  --train-sequences "$TRAIN_SEQ" \
  --output "$OUT_DIR/h1_stat_tests_1p7b_vs_4b.json"

step "11/11 stat_tests 4B vs 8B" "$STAT" \
  --eval-1p7b "$OUT_DIR/eval_4b.json" \
  --eval-8b   "$OUT_DIR/eval_8b.json" \
  --catalog "$CATALOG" \
  --train-sequences "$TRAIN_SEQ" \
  --output "$OUT_DIR/h1_stat_tests_4b_vs_8b.json"

echo
echo "DONE. Results in $OUT_DIR/:"
ls -la "$OUT_DIR/"
