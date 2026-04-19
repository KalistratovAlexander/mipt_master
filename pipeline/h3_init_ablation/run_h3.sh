#!/bin/bash
set -euo pipefail

# Reduce CUDA memory fragmentation across sequential arms — without this,
# arm_A_seed_43 Stage 1 hit OOM at step 77/2000 on a freshly-freed H100
# (vocab projection (64, s, 152696) fp32 = ~20GiB transient allocation).
# Arm-symmetric: applied identically to all 12 (arm, seed) combos.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# H3 init-ablation runner — one (arm, seed) combo.
#
# Assumes layout produced by pack_h3.sh:
#   /workspace/
#     data/  stage1/  stage2/  h3_init_ablation/  setup.sh
#
# Usage (on vast.ai):
#   bash /workspace/h3_init_ablation/run_h3.sh <ARM> <SEED>
#
# Pre-reqs:
#   1. artifacts/h3_init_scales.json has non-null target_frobenius_ctrl AND
#      target_frobenius_sid (run precompute_all.py before the first training run).
#   2. For arm C: artifacts/title_token_ids_per_sid.json exists.
#   3. For arm D: artifacts/codebook.pt exists.

ARM="${1:?arm required: A|B|C|D}"
SEED="${2:?seed required: int}"

WORKSPACE="${WORKSPACE:-/workspace}"
H3_DIR="$WORKSPACE/h3_init_ablation"
DATA_DIR="$WORKSPACE/data"
SCALES_JSON="$H3_DIR/artifacts/h3_init_scales.json"
RUN_DIR="$H3_DIR/runs/arm_${ARM}_seed_${SEED}"
MODEL_NAME="${H3_MODEL_NAME:-Qwen/Qwen3-0.6B}"

# DRY_RUN=1 → ~10 min smoke test with tiny samples/steps. Default = full run.
DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" == "1" ]]; then
    echo ">>> DRY RUN mode — small samples/steps for pipeline smoke test"
    S1_TRAIN_SAMPLES=1000;      S1_VAL_SAMPLES=100; S1_MAX_STEPS=50
    S2_TRAIN_SAMPLES=1000;      S2_SNAPSHOT_STEPS=10; S2_MAX_SNAPSHOTS=2
    EVAL_N_SAMPLES=10
else
    S1_TRAIN_SAMPLES=64000;     S1_VAL_SAMPLES=2000; S1_MAX_STEPS=2000
    S2_TRAIN_SAMPLES=1280000;   S2_SNAPSHOT_STEPS=1500; S2_MAX_SNAPSHOTS=6
    EVAL_N_SAMPLES=1000
fi

mkdir -p "$RUN_DIR"

# Pull pre-registered per-block Frobenius targets ----------------------------
read -r TARGET_CTRL TARGET_SID < <(python3 -c "
import json, sys
d = json.load(open('$SCALES_JSON'))
tc, ts = d.get('target_frobenius_ctrl'), d.get('target_frobenius_sid')
if tc is None or ts is None:
    sys.exit('target_frobenius_ctrl/sid missing in $SCALES_JSON — run precompute_all.py first')
print(tc, ts)
")
echo ">>> arm=$ARM seed=$SEED model=$MODEL_NAME  target_ctrl=$TARGET_CTRL  target_sid=$TARGET_SID"

# Arm-specific extra flags ----------------------------------------------------
ARM_EXTRA=()
if [[ "$ARM" == "C" ]]; then
    TITLE_MAP="$H3_DIR/artifacts/title_token_ids_per_sid.json"
    [[ -f "$TITLE_MAP" ]] || { echo "ERROR: $TITLE_MAP missing (required for arm C)"; exit 1; }
    ARM_EXTRA+=(--title-map-path "$TITLE_MAP")
fi
if [[ "$ARM" == "D" ]]; then
    CODEBOOK="$H3_DIR/artifacts/codebook.pt"
    [[ -f "$CODEBOOK" ]] || { echo "ERROR: $CODEBOOK missing (required for arm D)"; exit 1; }
    ARM_EXTRA+=(--rqvae-codebook-path "$CODEBOOK")
fi

STAGE1_OUT="$RUN_DIR/stage1"
STAGE2_OUT="$RUN_DIR/stage2"

# --- Setup deps (shared between stages) -------------------------------------
# shellcheck source=/dev/null
[[ -f "$WORKSPACE/setup.sh" ]] && source "$WORKSPACE/setup.sh"

# --- Stage 1 (skip if already done) -----------------------------------------
if [[ -f "$STAGE1_OUT/final/config.json" ]]; then
    echo ">>> [$(date +%T)] Stage 1 — SKIP (found $STAGE1_OUT/final)"
else
    echo ">>> [$(date +%T)] Stage 1 — vocab expansion"
    python3 "$WORKSPACE/stage1/train_1.8b.py" \
        --model-name "$MODEL_NAME" \
        --train-file "$DATA_DIR/semantic_llm_training/Pet_Supplies_conversations_train.parquet" \
        --val-file "$DATA_DIR/semantic_llm_training/Pet_Supplies_conversations_val.parquet" \
        --output-dir "$STAGE1_OUT" \
        --max-seq-length 512 --max-train-samples "$S1_TRAIN_SAMPLES" --max-val-samples "$S1_VAL_SAMPLES" \
        --lr 1e-3 --batch-size 64 --grad-accum 1 --max-steps "$S1_MAX_STEPS" --warmup-steps 100 \
        --logging-steps 50 --eval-steps 250 --save-steps 500 --no-wandb \
        --seed "$SEED" \
        --init-strategy "$ARM" \
        --init-seed "$SEED" \
        --target-frobenius-ctrl "$TARGET_CTRL" \
        --target-frobenius-sid "$TARGET_SID" \
        --h3-module-path "$H3_DIR" \
        "${ARM_EXTRA[@]}" \
        2>&1 | tee "$RUN_DIR/stage1.log"
fi

# --- Stage 2 (skip if already done) -----------------------------------------
if [[ -f "$STAGE2_OUT/final/config.json" ]]; then
    echo ">>> [$(date +%T)] Stage 2 — SKIP (found $STAGE2_OUT/final)"
else
    echo ">>> [$(date +%T)] Stage 2 — full fine-tune"
    python3 "$WORKSPACE/stage2/train_1.8b.py" \
        --stage1-model "$STAGE1_OUT/final" \
        --train-file "$DATA_DIR/semantic_llm_training/Pet_Supplies_conversations_train.parquet" \
        --val-file "$DATA_DIR/semantic_llm_training/Pet_Supplies_conversations_val.parquet" \
        --output-dir "$STAGE2_OUT" \
        --max-seq-length 512 --lr 2e-5 --batch-size 64 --grad-accum 2 \
        --epochs 1 --max-train-samples "$S2_TRAIN_SAMPLES" \
        --warmup-ratio 0.03 --weight-decay 0.01 --packing \
        --snapshot-steps "$S2_SNAPSHOT_STEPS" --max-snapshots "$S2_MAX_SNAPSHOTS" \
        --eval-steps 500 --sid-eval-samples 200 --logging-steps 25 --no-wandb \
        --seed "$SEED" \
        2>&1 | tee "$RUN_DIR/stage2.log"
fi

# --- Eval 1: primary (pre-registered) — Recall@10 on title→SID --------------
# Writes per-sample hit@10 array consumed by aggregate_stats.py paired bootstrap.
# FA2 + batched decode + early-stop on <|sid_end|> → ~5-10× speedup vs default.
echo ">>> [$(date +%T)] Evaluating primary Recall@10 (title_to_sid)"
python3 "$H3_DIR/evaluate_recall_at_10.py" \
    --model-path "$STAGE2_OUT/final" \
    --val-file "$DATA_DIR/semantic_llm_training/Pet_Supplies_conversations_val.parquet" \
    --n-samples "$EVAL_N_SAMPLES" --beam-size 10 --seed 42 \
    --attn-impl flash_attention_2 --max-new-tokens 16 --batch-size 16 \
    --output "$RUN_DIR/results.json" \
    2>&1 | tee "$RUN_DIR/eval_primary.log"

# --- Eval 2: descriptive — all 11 tasks via evaluate_unified.py -------------
# Covers Text→SID × 3, Sequential × 3, Co-purchase × 2, SID→Text × 3, WikiText PPL.
# SID tasks keep N=$EVAL_N_SAMPLES (pre-registered); text tasks use N=200
# (binomial CI at N=200 is still tight enough for descriptive ranking).
echo ">>> [$(date +%T)] Evaluating unified (11 tasks + WikiText-2 PPL)"
python3 "$WORKSPACE/evaluation/evaluate_unified.py" \
    --model-path "$STAGE2_OUT/final" \
    --data-dir "$DATA_DIR" \
    --model-name "arm_${ARM}_seed_${SEED}" \
    --samples-per-task "$EVAL_N_SAMPLES" --samples-per-task-text 200 \
    --beam-size 10 --seed 42 \
    --attn-impl flash_attention_2 --max-new-tokens-sid 16 --sid-batch-size 8 \
    --skip-benchmark \
    --output "$RUN_DIR/results_unified.json" \
    2>&1 | tee "$RUN_DIR/eval_unified.log"

# --- Eval 3: learning curve — Recall@10 on title_to_sid per snapshot --------
# 6 snapshots × N=1000 per §3.2.7 of thesis/h3_metrics_plan.md.
echo ">>> [$(date +%T)] Evaluating learning curve (Recall@10 per snapshot)"
mkdir -p "$RUN_DIR/learning_curve"
for snap in "$STAGE2_OUT/snapshots"/step-*; do
    [[ -d "$snap" ]] || continue
    step=$(basename "$snap" | sed 's/^step-//')
    echo ">>> [$(date +%T)]  snapshot step=$step"
    python3 "$H3_DIR/evaluate_recall_at_10.py" \
        --model-path "$snap" \
        --val-file "$DATA_DIR/semantic_llm_training/Pet_Supplies_conversations_val.parquet" \
        --n-samples "$EVAL_N_SAMPLES" --beam-size 10 --seed 42 \
        --attn-impl flash_attention_2 --max-new-tokens 16 --batch-size 16 \
        --output "$RUN_DIR/learning_curve/step_${step}.json" \
        2>&1 | tee -a "$RUN_DIR/eval_learning_curve.log"
done

echo ">>> [$(date +%T)] arm=$ARM seed=$SEED complete → $RUN_DIR/"
