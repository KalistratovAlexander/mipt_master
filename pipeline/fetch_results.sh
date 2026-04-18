#!/bin/bash
set -euo pipefail

# Pull experiment results (JSONs + logs only, no weights) from vast.ai to local repo.
#
# Usage:
#   bash pipeline/fetch_results.sh <HOST> <PORT> [EXPERIMENT]
#   VAST_HOST=... VAST_PORT=... bash pipeline/fetch_results.sh
#
# EXPERIMENT defaults to h3_init_ablation (must match remote /workspace/<EXPERIMENT>
# and local pipeline/<EXPERIMENT>).

HOST="${1:-${VAST_HOST:?host required: arg1 or VAST_HOST}}"
PORT="${2:-${VAST_PORT:?port required: arg2 or VAST_PORT}}"
EXPERIMENT="${3:-h3_init_ablation}"

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)/$EXPERIMENT"
REMOTE="root@$HOST:/workspace/$EXPERIMENT"

echo ">>> Pulling $EXPERIMENT from $HOST:$PORT"

rsync -avz -e "ssh -p $PORT" \
    --include='*/' \
    --include='results.json' \
    --include='results_unified.json' \
    --include='learning_curve/***' \
    --include='*.log' \
    --exclude='*' \
    "$REMOTE/runs/" "$LOCAL_DIR/runs/"

rsync -avz -e "ssh -p $PORT" \
    --include='*.json' \
    --exclude='*' \
    "$REMOTE/results/" "$LOCAL_DIR/results/"

echo ">>> Done. Local: $LOCAL_DIR/{runs,results}/"
