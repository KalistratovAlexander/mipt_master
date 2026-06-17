#!/bin/bash
set -euo pipefail

# Smoke test: Stage 1 (100 steps) + Stage 2 (100 steps) — ~10 min total on H100.
#
# Checks:
#   - Pipeline runs end-to-end without errors
#   - No NaN loss (TrainingMonitorCallback raises RuntimeError if detected)
#   - SID embeddings are learning (EmbeddingMonitorCallback logs norms)
#   - SIDEvalCallback fires and produces SID predictions
#   - Checkpoint saving works
#   - HF push is skipped (SMOKE_TEST=1)
#
# Usage (from /workspace after extracting the package):
#   bash run_smoke.sh
#
# Check logs:
#   tail -f stage1/train.log
#   tail -f stage2/train.log

export SMOKE_TEST=1

echo "=========================================="
echo "SMOKE TEST START: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

echo ""
echo "=== Stage 1: vocab expansion (100 steps, 1000 samples) ==="
bash stage1/run.sh \
    --max-steps 100 \
    --max-train-samples 1000 \
    --max-val-samples 200 \
    --logging-steps 10 \
    --eval-steps 50 \
    --save-steps 50

echo ""
echo "=== Stage 2: full finetune (100 steps, 500 samples) ==="
bash stage2/run.sh \
    --max-steps 100 \
    --max-train-samples 500 \
    --max-val-samples 200 \
    --logging-steps 10 \
    --eval-steps 50 \
    --snapshot-steps 9999 \
    --sid-eval-samples 20

echo ""
echo "=========================================="
echo "SMOKE TEST PASSED: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Logs: stage1/train.log  stage2/train.log"
echo "=========================================="
