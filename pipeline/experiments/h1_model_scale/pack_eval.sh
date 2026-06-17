#!/bin/bash
set -euo pipefail

# Pack vast.ai evaluation package for H1 evaluation.
# Includes: eval scripts + all required data files (~180MB compressed).
#
# Usage: cd mipt_master && bash pipeline/experiments/h1_model_scale/pack_eval.sh
#
# Then on eval machine (H100 80GB, 100GB+ disk):
#   cd /workspace && tar xf vast_eval_package.tar.gz
#   export HF_TOKEN=hf_...   # needed for model download from HF
#   bash pipeline/evaluation/run_h1.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT="$PROJECT_DIR/vast_eval_package.tar.gz"

echo ">>> Building H1 eval package..."

# --- Check required data files ---
check_file() { [ -f "$1" ] || { echo "ERROR: missing $1"; exit 1; }; }
check_file "$PROJECT_DIR/data/semantic_llm_training/Pet_Supplies_conversations_val.parquet"
check_file "$PROJECT_DIR/data/embeds/Pet_Supplies_items_with_semantic_ids.parquet"
check_file "$PROJECT_DIR/data/embeds/Pet_Supplies_items_with_embeddings_with_semantic_ids.parquet"
check_file "$PROJECT_DIR/data/sequences/Pet_Supplies_sequences_with_sid_train.parquet"

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

echo "  Copying eval scripts..."
mkdir -p "$TMP/pipeline/evaluation"
cp "$PROJECT_DIR/pipeline/evaluation/evaluate_unified.py" "$TMP/pipeline/evaluation/"
cp "$SCRIPT_DIR/stat_tests.py"       "$TMP/pipeline/evaluation/"
cp "$SCRIPT_DIR/run_h1.sh"           "$TMP/pipeline/evaluation/"
cp "$SCRIPT_DIR/setup_eval.sh"       "$TMP/pipeline/evaluation/"

echo "  Copying data..."
mkdir -p "$TMP/data/semantic_llm_training" "$TMP/data/embeds" "$TMP/data/sequences"
cp "$PROJECT_DIR/data/semantic_llm_training/Pet_Supplies_conversations_val.parquet" \
   "$TMP/data/semantic_llm_training/"
cp "$PROJECT_DIR/data/embeds/Pet_Supplies_items_with_semantic_ids.parquet" \
   "$TMP/data/embeds/"
cp "$PROJECT_DIR/data/embeds/Pet_Supplies_items_with_embeddings_with_semantic_ids.parquet" \
   "$TMP/data/embeds/"
cp "$PROJECT_DIR/data/sequences/Pet_Supplies_sequences_with_sid_train.parquet" \
   "$TMP/data/sequences/"

echo "  Compressing..."
tar -czf "$OUT" -C "$TMP" .

SIZE=$(du -h "$OUT" | cut -f1)
echo ">>> Done: $OUT ($SIZE)"
echo ""
echo "Upload to eval machine:"
echo "  scp -P <PORT> $OUT root@<HOST>:/workspace/"
echo ""
echo "Then on server:"
echo "  cd /workspace && tar xf vast_eval_package.tar.gz"
echo "  export HF_TOKEN=hf_..."
echo "  bash pipeline/evaluation/run_h1.sh"
