#!/bin/bash
set -euo pipefail

# Pack vast.ai training package for Qwen3 small models (0.6B / 1.8B / 4B)
# Creates vast_<MODEL>_package.tar.gz with the correct /workspace structure
#
# Usage: cd mipt_master && bash pipeline/fine_tune_h1/pack.sh [0.6b|1.8b|4b]
# Default: 1.8b

MODEL="${1:-1.8b}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$PROJECT_DIR/vast_${MODEL}_package.tar.gz"

case "$MODEL" in
    0.6b) RUN_S1="run_0.6b.sh"; RUN_S2="run_0.6b.sh" ;;
    1.8b) RUN_S1="run_1.8b.sh"; RUN_S2="run_1.8b.sh" ;;
    4b)   RUN_S1="run_4b.sh";   RUN_S2="run_4b.sh"   ;;
    8b)   RUN_S1="run_8b.sh";   RUN_S2="run_8b.sh"   ;;
    *)    echo "ERROR: unknown model '$MODEL'. Use: 0.6b | 1.8b | 4b | 8b" && exit 1 ;;
esac

echo ">>> Building vast.ai package for Qwen3-${MODEL}..."

DATA_DIR="$PROJECT_DIR/data/semantic_llm_training"
[ ! -f "$DATA_DIR/Pet_Supplies_conversations_train.parquet" ] && echo "ERROR: train data not found at $DATA_DIR" && exit 1
[ ! -f "$DATA_DIR/Pet_Supplies_conversations_val.parquet" ] && echo "ERROR: val data not found at $DATA_DIR" && exit 1

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

echo "  Copying data..."
mkdir -p "$TMP/data"
cp "$DATA_DIR/Pet_Supplies_conversations_train.parquet" "$TMP/data/"
cp "$DATA_DIR/Pet_Supplies_conversations_val.parquet" "$TMP/data/"

echo "  Copying Stage 1..."
mkdir -p "$TMP/stage1"
cp "$SCRIPT_DIR/stage1_vocab_expansion/train_1.8b.py" "$TMP/stage1/"
cp "$SCRIPT_DIR/stage1_vocab_expansion/$RUN_S1" "$TMP/stage1/run.sh"

echo "  Copying Stage 2..."
mkdir -p "$TMP/stage2"
cp "$SCRIPT_DIR/stage2_full_finetune/train_1.8b.py" "$TMP/stage2/"
cp "$SCRIPT_DIR/stage2_full_finetune/$RUN_S2" "$TMP/stage2/run.sh"

echo "  Copying setup and smoke test..."
cp "$SCRIPT_DIR/setup.sh" "$TMP/"
cp "$SCRIPT_DIR/run_smoke.sh" "$TMP/"

echo "  Compressing..."
tar -czf "$OUT" -C "$TMP" .

SIZE=$(du -h "$OUT" | cut -f1)
echo ">>> Done: $OUT ($SIZE)"
echo ""
echo "Upload to vast.ai:"
echo "  scp -P <PORT> $OUT root@<HOST>:/workspace/"
echo ""
echo "Then on server:"
echo "  cd /workspace && tar xf vast_${MODEL}_package.tar.gz"
echo "  export HF_TOKEN=hf_..."
echo "  bash run_smoke.sh    # smoke test first (~10 min)"
echo "  bash stage1/run.sh   # Stage 1 full"
echo "  bash stage2/run.sh   # Stage 2 full"
