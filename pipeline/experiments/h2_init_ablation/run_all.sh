#!/bin/bash
set -euo pipefail

# Orchestrate all 12 H2 runs (4 arms × 3 seeds) + post-hoc diagnostics.
#
# Skip-if-done: a run is skipped when BOTH results.json and results_unified.json
# already exist for that (arm, seed). Delete either file to re-run.
#
# Respects DRY_RUN env var (passed through to run.sh). Example:
#   DRY_RUN=1 bash run_all.sh   # ~10 min smoke over all 12
#   bash run_all.sh             # full run (≈ full-scale training budget)

WORKSPACE="${WORKSPACE:-/workspace}"
H2_DIR="$WORKSPACE/h2_init_ablation"
RUNS_DIR="$H2_DIR/runs"

ARMS=(A B C D)
SEEDS=(42 43 44)

START_TS=$(date +%s)
TOTAL=$(( ${#ARMS[@]} * ${#SEEDS[@]} ))
N=0

for arm in "${ARMS[@]}"; do
    for sd in "${SEEDS[@]}"; do
        N=$(( N + 1 ))
        RUN_DIR="$RUNS_DIR/arm_${arm}_seed_${sd}"
        R1="$RUN_DIR/results.json"
        R2="$RUN_DIR/results_unified.json"
        if [[ -f "$R1" && -f "$R2" ]]; then
            echo ">>> [$N/$TOTAL] arm=$arm seed=$sd — SKIP (results exist)"
            continue
        fi
        echo ">>> [$N/$TOTAL] arm=$arm seed=$sd — running"
        bash "$H2_DIR/run.sh" "$arm" "$sd"
    done
done

# --- Transversal diagnostics (geometry) -------------------------------------
echo ">>> [post-hoc] transversal_diagnostics.py"
python3 "$H2_DIR/transversal_diagnostics.py" \
    --runs-dir "$RUNS_DIR" \
    --codebook-path "$H2_DIR/artifacts/codebook.pt" \
    --output "$H2_DIR/results/transversal.json"

# --- Aggregate stats (primary hypothesis test + descriptive surface) --------
echo ">>> [post-hoc] aggregate_stats.py"
python3 "$H2_DIR/aggregate_stats.py" \
    --runs-dir "$RUNS_DIR" \
    --output "$H2_DIR/results/h2_summary.json"

ELAPSED=$(( $(date +%s) - START_TS ))
echo ">>> All done in $((ELAPSED / 60)) min → $H2_DIR/results/"
