#!/usr/bin/env python3
"""Partial aggregation of H3 runs — tolerant to missing (arm, seed) combos.

Sibling of aggregate_stats.py for mid-run peeks. Reports whatever it can
given the (arm, seed) combos that have finished, and notes what's still
pending for the pre-registered analysis.

Does NOT replace aggregate_stats.py — the final registered analysis must
wait for the full 12-run grid. This is diagnostic only.

Usage:
    python partial_aggregate.py --runs-dir /workspace/h3_init_ablation/runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARMS = ["A", "B", "C", "D"]
SEEDS = [42, 43, 44]
FRIEDMAN_ARMS = ["A", "C", "D"]            # pre-reg: B is control
PLANNED_CONTRASTS = [("A", "C"), ("A", "D"), ("C", "D")]
ALPHA_CORRECTED = 0.05 / len(PLANNED_CONTRASTS)


def _safe_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _paired_bootstrap(hits_x, hits_y, n_bootstrap, alpha, rng):
    assert hits_x.shape == hits_y.shape
    diff = hits_x.astype(np.float64) - hits_y.astype(np.float64)
    obs = float(diff.mean())
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, len(hits_x), size=len(hits_x))
        boot[b] = diff[idx].mean()
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    p = 2.0 * min(float((boot >= 0).mean()), float((boot <= 0).mean()))
    return {"n": len(hits_x), "mean_diff": obs, "ci_lo": float(lo),
            "ci_hi": float(hi), "p_value": p, "sig": (lo > 0) or (hi < 0)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None,
                   help="Optional JSON dump path (stdout by default).")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    primary: dict = {arm: {} for arm in ARMS}
    unified: dict = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for sd in SEEDS:
            run_dir = runs_dir / f"arm_{arm}_seed_{sd}"
            primary[arm][sd] = _safe_load(run_dir / "results.json")
            unified[arm][sd] = _safe_load(run_dir / "results_unified.json")

    # --- Presence matrix ---
    print("Presence (primary / unified):")
    print("       " + "    ".join(f" seed={s}" for s in SEEDS))
    for arm in ARMS:
        row = [arm, "  "]
        for sd in SEEDS:
            p_ok = "●" if primary[arm][sd] else "·"
            u_ok = "●" if unified[arm][sd] else "·"
            row.append(f"{p_ok}/{u_ok}   ")
        print("".join(row))
    print("  (● = done · = pending)")

    # --- Per-arm recall@10 summary (primary) ---
    print("\nrecall@10 per (arm, seed) from primary eval:")
    summary = {}
    for arm in ARMS:
        vals = []
        per_seed = {}
        for sd in SEEDS:
            d = primary[arm][sd]
            if d is not None and "recall@10" in d:
                vals.append(d["recall@10"])
                per_seed[sd] = d["recall@10"]
        if vals:
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            summary[arm] = {"n_seeds": len(vals), "mean": mean, "std": std,
                            "per_seed": per_seed}
            std_str = f"±{std:.4f}" if std is not None else "(single seed)"
            print(f"  {arm}: n={len(vals)}  mean={mean:.4f} {std_str}  per_seed={per_seed}")
        else:
            summary[arm] = {"n_seeds": 0}
            print(f"  {arm}: no data yet")

    # --- WikiText-2 PPL per arm ---
    print("\nWikiText-2 PPL per arm:")
    ppl_summary = {}
    for arm in ARMS:
        vals = []
        for sd in SEEDS:
            u = unified[arm][sd]
            if u is not None:
                v = u.get("perplexity_wikitext2", {}).get("perplexity")
                if v is not None:
                    vals.append(v)
        if vals:
            ppl_summary[arm] = {"n_seeds": len(vals),
                                 "mean": float(np.mean(vals)),
                                 "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
            std_str = f"±{ppl_summary[arm]['std']:.2f}" if ppl_summary[arm]['std'] is not None else ""
            print(f"  {arm}: mean={ppl_summary[arm]['mean']:.2f} {std_str}  n={len(vals)}")
        else:
            ppl_summary[arm] = {"n_seeds": 0}
            print(f"  {arm}: no data yet")

    # --- Paired bootstrap on whatever pairs are available ---
    # For each planned contrast, intersect seeds where BOTH arms have primary.
    print("\nPaired bootstrap contrasts (on seeds present for BOTH arms):")
    rng = np.random.default_rng(args.seed)
    contrast_results = {}
    for x, y in PLANNED_CONTRASTS:
        common_seeds = [sd for sd in SEEDS
                        if primary[x][sd] is not None and primary[y][sd] is not None]
        if not common_seeds:
            print(f"  {x}_vs_{y}: pending (no seed has both arms done yet)")
            contrast_results[f"{x}_vs_{y}"] = {"status": "pending"}
            continue
        hits_x = np.concatenate([np.array(primary[x][sd]["per_sample_hit@10"])
                                 for sd in common_seeds])
        hits_y = np.concatenate([np.array(primary[y][sd]["per_sample_hit@10"])
                                 for sd in common_seeds])
        r = _paired_bootstrap(hits_x, hits_y, args.n_bootstrap, ALPHA_CORRECTED, rng)
        r["seeds_used"] = common_seeds
        contrast_results[f"{x}_vs_{y}"] = r
        sig_mark = "*" if r["sig"] else " "
        print(f"  {x}_vs_{y}: seeds={common_seeds}  Δ={r['mean_diff']:+.4f}  "
              f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  p={r['p_value']:.4f} {sig_mark}")

    # --- What's pending for full pre-reg analysis ---
    missing = []
    for arm in ARMS:
        for sd in SEEDS:
            if primary[arm][sd] is None or unified[arm][sd] is None:
                missing.append(f"arm_{arm}_seed_{sd}")
    print(f"\nPending for full pre-reg analysis: {len(missing)}/{len(ARMS)*len(SEEDS)}")
    if missing:
        print(f"  {missing}")

    # Friedman readiness
    min_n_per_friedman_arm = min(summary[a]["n_seeds"] for a in FRIEDMAN_ARMS)
    if min_n_per_friedman_arm >= 2:
        print(f"\nFriedman-ready: all of {FRIEDMAN_ARMS} have ≥2 seeds ({min_n_per_friedman_arm}).")
    else:
        print(f"\nFriedman NOT ready: {FRIEDMAN_ARMS} need ≥2 seeds each "
              f"(current min = {min_n_per_friedman_arm}).")

    out = {
        "summary": summary,
        "ppl": ppl_summary,
        "contrasts": contrast_results,
        "n_pending": len(missing),
        "pending": missing,
        "friedman_ready": min_n_per_friedman_arm >= 2,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
