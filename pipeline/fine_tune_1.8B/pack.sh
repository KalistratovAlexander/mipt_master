#!/bin/bash
set -euo pipefail

# Pack vast.ai training package
# Creates vast_training_package.tar.gz with the correct /workspace structure
#
# Usage: cd mipt_master && bash fine_tune_1.8B/pack.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUT="$PROJECT_DIR/vast_training_package.tar.gz"

echo ">>> Building vast.ai training package..."

# --- Verify source data exists ---
DATA_DIR="$PROJECT_DIR/data/semantic_llm_training"
[ ! -f "$DATA_DIR/Pet_Supplies_conversations_train.parquet" ] && echo "ERROR: train data not found at $DATA_DIR" && exit 1
[ ! -f "$DATA_DIR/Pet_Supplies_conversations_val.parquet" ] && echo "ERROR: val data not found at $DATA_DIR" && exit 1

# --- Build package in temp dir ---
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

echo "  Copying data..."
mkdir -p "$TMP/data"
cp "$DATA_DIR/Pet_Supplies_conversations_train.parquet" "$TMP/data/"
cp "$DATA_DIR/Pet_Supplies_conversations_val.parquet" "$TMP/data/"

echo "  Copying Stage 1..."
mkdir -p "$TMP/stage1"
cp "$SCRIPT_DIR/stage1_vocab_expansion/train_1.8b.py" "$TMP/stage1/"
cp "$SCRIPT_DIR/stage1_vocab_expansion/run_1.8b.sh" "$TMP/stage1/run.sh"

echo "  Copying Stage 2..."
mkdir -p "$TMP/stage2"
cp "$SCRIPT_DIR/stage2_full_finetune/train_1.8b.py" "$TMP/stage2/"
cp "$SCRIPT_DIR/stage2_full_finetune/run_1.8b.sh" "$TMP/stage2/run.sh"

echo "  Copying setup & README..."
cp "$SCRIPT_DIR/setup.sh" "$TMP/"
cp "$SCRIPT_DIR/README.md" "$TMP/"

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
echo "  cd /workspace && tar xf vast_training_package.tar.gz"
