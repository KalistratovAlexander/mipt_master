#!/bin/bash
set -euo pipefail

# Pack vast.ai training package for H2 init-ablation (Qwen3-0.6B)
# Creates h2_vast_package.tar.gz with the correct /workspace structure
#
# Usage: bash pipeline/h2_init_ablation/pack_h2.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
FT_DIR="$PROJECT_DIR/pipeline/fine_tune"
EVAL_DIR="$PROJECT_DIR/pipeline/evaluation"
OUT="$PROJECT_DIR/h2_vast_package.tar.gz"

echo ">>> Building H2 vast.ai package..."

# --- Verify source files exist ---
DATA_DIR="$PROJECT_DIR/data/semantic_llm_training"
ITEMS_CLEANED="$PROJECT_DIR/data/prepared/Pet_Supplies_items_cleaned.parquet"
ITEMS_WITH_SIDS="$PROJECT_DIR/data/embeds/Pet_Supplies_items_with_semantic_ids.parquet"
RQVAE_CKPT="$PROJECT_DIR/models/rqvae/best_model.pth"

for f in \
    "$DATA_DIR/Pet_Supplies_conversations_train.parquet" \
    "$DATA_DIR/Pet_Supplies_conversations_val.parquet" \
    "$ITEMS_CLEANED" "$ITEMS_WITH_SIDS" "$RQVAE_CKPT"
do
    [ ! -f "$f" ] && echo "ERROR: missing $f" && exit 1
done

# --- Build package in temp dir ---
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

echo "  Copying data..."
mkdir -p "$TMP/data/semantic_llm_training" "$TMP/data/prepared" "$TMP/data/embeds"
cp "$DATA_DIR/Pet_Supplies_conversations_train.parquet" "$TMP/data/semantic_llm_training/"
cp "$DATA_DIR/Pet_Supplies_conversations_val.parquet"   "$TMP/data/semantic_llm_training/"
cp "$ITEMS_CLEANED"   "$TMP/data/prepared/"
cp "$ITEMS_WITH_SIDS" "$TMP/data/embeds/"

echo "  Copying RQ-VAE checkpoint..."
mkdir -p "$TMP/models/rqvae"
cp "$RQVAE_CKPT" "$TMP/models/rqvae/"

echo "  Copying training scripts..."
mkdir -p "$TMP/stage1" "$TMP/stage2"
cp "$FT_DIR/stage1_vocab_expansion/train_1.8b.py" "$TMP/stage1/"
cp "$FT_DIR/stage2_full_finetune/train_1.8b.py"   "$TMP/stage2/"
cp "$FT_DIR/setup.sh" "$TMP/"

echo "  Copying shared evaluator..."
mkdir -p "$TMP/evaluation"
cp "$EVAL_DIR/evaluate_unified.py" "$TMP/evaluation/"

echo "  Copying H2 module..."
mkdir -p "$TMP/h2_init_ablation"
cp "$SCRIPT_DIR"/*.py "$TMP/h2_init_ablation/"
cp "$SCRIPT_DIR"/*.sh "$TMP/h2_init_ablation/"
cp -r "$SCRIPT_DIR/artifacts" "$TMP/h2_init_ablation/"
mkdir -p "$TMP/h2_init_ablation/runs" "$TMP/h2_init_ablation/results"

# --- Create archive ---
echo "  Compressing..."
tar -czf "$OUT" -C "$TMP" .

SIZE=$(du -h "$OUT" | cut -f1)
echo ">>> Done: $OUT ($SIZE)"
echo ""
echo "Upload to vast.ai:"
echo "  scp -P <PORT> $OUT root@<HOST>:/workspace/"
echo ""
echo "Then on server:"
echo "  cd /workspace && tar xf h2_vast_package.tar.gz"
echo "  cd /workspace && python3 h2_init_ablation/precompute_all.py  # one-shot pre-reg"
echo "  DRY_RUN=1 bash h2_init_ablation/run_h2.sh A 42   # smoke test (~10 min)"
echo "  bash h2_init_ablation/run_all.sh                 # 12 runs + diagnostics"
