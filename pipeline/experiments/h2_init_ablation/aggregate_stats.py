#!/usr/bin/env python3
"""H2 final statistics: primary hypothesis test + descriptive surface.

Walks runs/arm_{A,B,C,D}_seed_{42,43,44}/ and collects:
  1. results.json           — primary metric (Recall@10 on title_to_sid)
                               from evaluate_recall_at_10.py, with per-sample hit@10.
  2. results_unified.json   — all 11 tasks + WikiText-2 PPL from evaluate_unified.py.
  3. learning_curve/*.json  — 6 snapshots × Recall@10.

Primary test (thesis §3.2.3 / §3.2.5):
  - 3 paired bootstrap contrasts {A-C, A-D, C-D} on concatenated per-sample hits.
  - Bonferroni correction over m=3: alpha_corrected = 0.05 / 3 = 0.01667.
  Arm B is a variance-matched random control, reported descriptively only.

Descriptive surface:
  - per-arm × per-task Recall@10 / NDCG@10 / hier_hit@10 / valid-format means (±std).
  - per-arm WikiText-2 PPL (mean ± std).
  - per-arm learning-curve trace (step, recall@10) aggregated over seeds.
  - transversal diagnostics (cos / CKA / effective rank / RSA) from
    results/transversal.json if present.

Arm B is descriptive-only (variance-matched random control).

Writes results/h2_summary.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from init_strategies import load_pre_registered  # noqa: E402


def _load_run(runs_dir: Path, arm: str, seed: int) -> dict:
    path = runs_dir / f"arm_{arm}_seed_{seed}" / "results.json"
    if not path.exists():
        raise SystemExit(f"Missing run: {path}")
    with open(path) as f:
        return json.load(f)


def _load_unified(runs_dir: Path, arm: str, seed: int) -> dict | None:
    path = runs_dir / f"arm_{arm}_seed_{seed}" / "results_unified.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_learning_curve(runs_dir: Path, arm: str, seed: int) -> list[dict]:
    """Return [{"step": N, "recall@10": x, "recall@1": x, "recall@5": x}, ...] sorted by step."""
    lc_dir = runs_dir / f"arm_{arm}_seed_{seed}" / "learning_curve"
    if not lc_dir.exists():
        return []
    points = []
    for f_path in lc_dir.glob("step_*.json"):
        m = re.match(r"step_(\d+)\.json$", f_path.name)
        if not m:
            continue
        with open(f_path) as f:
            d = json.load(f)
        points.append({
            "step": int(m.group(1)),
            "recall@1": d.get("recall@1"),
            "recall@5": d.get("recall@5"),
            "recall@10": d.get("recall@10"),
        })
    points.sort(key=lambda p: p["step"])
    return points


def _paired_bootstrap(
    hits_x: np.ndarray,
    hits_y: np.ndarray,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict:
    """Paired bootstrap on per-sample 0/1 hits. Returns CI + p-value for mean diff."""
    assert hits_x.shape == hits_y.shape, f"{hits_x.shape} vs {hits_y.shape}"
    n = len(hits_x)
    diff = hits_x.astype(np.float64) - hits_y.astype(np.float64)
    observed = float(diff.mean())

    boot = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[b] = diff[idx].mean()

    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    # Two-sided p via achieved significance level
    p_two_sided = min(1.0, 2.0 * min(float((boot >= 0).mean()), float((boot <= 0).mean())))
    return {
        "mean_diff": observed,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_value": p_two_sided,
        "significant": bool((lo > 0) or (hi < 0)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default=str(Path(__file__).parent / "runs"))
    p.add_argument("--artifacts-path",
                   default=str(Path(__file__).parent / "artifacts" / "h2_init_scales.json"))
    p.add_argument("--output",
                   default=str(Path(__file__).parent / "results" / "h2_summary.json"))
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    proto = load_pre_registered(args.artifacts_path)
    arms = proto["arms"]                              # ["A", "B", "C", "D"]
    seeds = proto["seeds"]                            # [42, 43, 44]
    contrasts = proto["statistical_protocol"]["planned_contrasts"]  # [[A,C], [A,D], [C,D]]
    alpha_corrected = proto["statistical_protocol"]["alpha_corrected"]

    runs_dir = Path(args.runs_dir)
    data: dict[str, dict[int, dict]] = {arm: {} for arm in arms}
    for arm in arms:
        for sd in seeds:
            data[arm][sd] = _load_run(runs_dir, arm, sd)

    # --- 1. Per-arm, per-seed recall@10 table ---
    recall_table = {arm: [data[arm][sd]["recall@10"] for sd in seeds] for arm in arms}
    print("recall@10 per (arm, seed):")
    for arm in arms:
        print(f"  {arm}: {recall_table[arm]}  mean={np.mean(recall_table[arm]):.4f}")

    # --- 2. Planned paired bootstrap contrasts on concatenated hits ---
    rng = np.random.default_rng(args.seed)
    contrast_results = {}
    for pair in contrasts:
        x, y = pair
        hits_x = np.concatenate([np.array(data[x][sd]["per_sample_hit@10"]) for sd in seeds])
        hits_y = np.concatenate([np.array(data[y][sd]["per_sample_hit@10"]) for sd in seeds])
        r = _paired_bootstrap(hits_x, hits_y, args.n_bootstrap, alpha_corrected, rng)
        key = f"{x}_vs_{y}"
        contrast_results[key] = r
        print(
            f"{key:8s}: Δ={r['mean_diff']:+.4f}  "
            f"CI_{1 - alpha_corrected:.3%}=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
            f"p={r['p_value']:.4f}  sig={r['significant']}"
        )

    # --- 4. Descriptive surface from results_unified.json ---
    # Per (arm, seed) → raw unified.tasks dict. Per (arm, task) → mean/std of the
    # key recall@10 metric across seeds (downstream tooling reads deeper nested keys as needed).
    unified_by_run: dict = {}
    wikitext_by_run: dict = {}
    for arm in arms:
        unified_by_run[arm] = {}
        wikitext_by_run[arm] = {}
        for sd in seeds:
            u = _load_unified(runs_dir, arm, sd)
            if u is None:
                continue
            unified_by_run[arm][sd] = u.get("tasks", {})
            wikitext_by_run[arm][sd] = u.get("perplexity_wikitext2", {})

    wikitext_per_arm: dict = {}
    for arm in arms:
        ppls = [wikitext_by_run[arm].get(sd, {}).get("perplexity")
                for sd in seeds]
        ppls = [x for x in ppls if x is not None]
        wikitext_per_arm[arm] = {
            "mean": float(np.mean(ppls)) if ppls else None,
            "std": float(np.std(ppls, ddof=1)) if len(ppls) > 1 else None,
            "per_seed": [wikitext_by_run[arm].get(sd, {}).get("perplexity") for sd in seeds],
        }

    # --- 5. Learning-curve traces per (arm, seed), + per-arm aggregated ---
    learning_curve_by_run: dict = {}
    for arm in arms:
        learning_curve_by_run[arm] = {}
        for sd in seeds:
            learning_curve_by_run[arm][sd] = _load_learning_curve(runs_dir, arm, sd)

    # Aggregate per arm: step → [recalls across seeds]
    learning_curve_per_arm: dict = {}
    for arm in arms:
        step_to_recalls: dict[int, list[float]] = {}
        for sd in seeds:
            for pt in learning_curve_by_run[arm].get(sd, []):
                if pt["recall@10"] is None:
                    continue
                step_to_recalls.setdefault(pt["step"], []).append(pt["recall@10"])
        learning_curve_per_arm[arm] = [
            {
                "step": step,
                "recall@10_mean": float(np.mean(vals)),
                "recall@10_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "n_seeds": len(vals),
            }
            for step, vals in sorted(step_to_recalls.items())
        ]

    # --- 6. Transversal diagnostics (if available) ---
    transversal_path = Path(args.output).parent / "transversal.json"
    transversal = None
    if transversal_path.exists():
        with open(transversal_path) as f:
            transversal = json.load(f)

    summary = {
        "protocol": {
            "arms": arms,
            "seeds": seeds,
            "primary_metric": proto["statistical_protocol"]["primary_metric"],
            "contrasts": contrasts,
            "bonferroni_m": proto["statistical_protocol"]["bonferroni_m"],
            "alpha_corrected": alpha_corrected,
            "n_bootstrap": args.n_bootstrap,
            "target_frobenius": proto["target_frobenius"],
            "committed_git_sha": proto["committed_git_sha"],
        },
        "primary": {
            "per_arm_recall@10": recall_table,
            "per_arm_mean": {arm: float(np.mean(recall_table[arm])) for arm in arms},
            "per_arm_std": {arm: float(np.std(recall_table[arm], ddof=1)) for arm in arms},
            "contrasts": contrast_results,
        },
        "descriptive": {
            "unified_by_run": unified_by_run,
            "wikitext_ppl_per_arm": wikitext_per_arm,
            "learning_curve_by_run": learning_curve_by_run,
            "learning_curve_per_arm": learning_curve_per_arm,
        },
        "transversal": transversal,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")
    if transversal is None:
        print("NOTE: transversal.json not found — run transversal_diagnostics.py "
              "and re-run this script to include geometry diagnostics.")


if __name__ == "__main__":
    main()
