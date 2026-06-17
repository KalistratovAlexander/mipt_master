#!/usr/bin/env python3
"""Post-hoc B.7: catalog-hit rate per task, per arm.

For every SID task and every beam prediction, check if the predicted
(A,B,C,D) tuple is in the catalog set. Reports:
  - fraction of beam predictions that are catalog-valid (of 10 beams × N samples)
  - fraction of samples with at least one catalog-valid beam

Catalog = union of all `per_sample_gold` tuples across all 12 runs × 8 tasks.
Requires per_sample_beam_preds + per_sample_gold in results_unified.json
(Option-2 patch; pre-patch runs will be re-evaluated in reeval_pre_patch.sh).

Writes JSON: results/post_hoc_catalog_hit.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SID_TASKS = [
    "copurchase_backward", "copurchase_forward",
    "description_to_sid", "features_to_sid",
    "seq_last_2", "seq_last_3", "seq_last_5",
    "title_to_sid",
]


def load_unified(runs_dir: Path, arm: str, seed: int):
    p = runs_dir / f"arm_{arm}_seed_{seed}" / "results_unified.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def build_catalog(runs_data) -> set:
    """Union of all per_sample_gold SID tuples across all runs × tasks."""
    cat = set()
    for arm_seed, u in runs_data.items():
        for task in SID_TASKS:
            t = u.get("tasks", {}).get(task, {})
            beam = t.get("beam", {})
            gold = beam.get("per_sample_gold") or []
            for g in gold:
                if g is None:
                    continue
                # g is a list of 4 ints (A,B,C,D)
                cat.add(tuple(g))
    return cat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--arms", default="A,B,C,D")
    p.add_argument("--seeds", default="42,43,44")
    args = p.parse_args()

    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    runs_dir = Path(args.runs_dir)

    runs_data = {}
    missing = []
    for arm in arms:
        for sd in seeds:
            u = load_unified(runs_dir, arm, sd)
            if u is None:
                missing.append(f"arm_{arm}_seed_{sd}")
                continue
            runs_data[f"{arm}_{sd}"] = u

    if missing:
        print(f"WARN: missing {len(missing)} runs: {missing}")

    catalog = build_catalog(runs_data)
    print(f"Catalog size: {len(catalog)} unique SID tuples")

    # Per (arm, seed, task): beam_catalog_hit (0..1) and sample_catalog_hit (0..1)
    out = {
        "catalog_size": len(catalog),
        "missing_runs": missing,
        "per_task": {},
        "per_arm": {},
    }

    for task in SID_TASKS:
        out["per_task"][task] = {}
        for arm in arms:
            per_seed_beam = []
            per_seed_sample = []
            per_seed_valid_fmt = []
            for sd in seeds:
                u = runs_data.get(f"{arm}_{sd}")
                if u is None:
                    continue
                t = u.get("tasks", {}).get(task, {})
                beam = t.get("beam", {})
                preds = beam.get("per_sample_beam_preds")
                if not preds:
                    # pre-patch run; skip
                    continue
                n_pred_total = 0
                n_pred_in_cat = 0
                n_samples = 0
                n_samples_any = 0
                for beam_list in preds:
                    if not beam_list:
                        continue
                    n_samples += 1
                    any_hit = False
                    for p_ in beam_list:
                        if p_ is None:
                            continue
                        n_pred_total += 1
                        if tuple(p_) in catalog:
                            n_pred_in_cat += 1
                            any_hit = True
                    if any_hit:
                        n_samples_any += 1
                if n_pred_total:
                    per_seed_beam.append(n_pred_in_cat / n_pred_total)
                if n_samples:
                    per_seed_sample.append(n_samples_any / n_samples)
                g = beam.get("valid_format")
                if g is not None:
                    per_seed_valid_fmt.append(g)
            out["per_task"][task][arm] = {
                "beam_catalog_hit_mean": statistics.mean(per_seed_beam) if per_seed_beam else None,
                "beam_catalog_hit_std": statistics.stdev(per_seed_beam) if len(per_seed_beam) >= 2 else None,
                "sample_any_catalog_hit_mean": statistics.mean(per_seed_sample) if per_seed_sample else None,
                "sample_any_catalog_hit_std": statistics.stdev(per_seed_sample) if len(per_seed_sample) >= 2 else None,
                "n_seeds": len(per_seed_beam),
                "valid_format_mean": statistics.mean(per_seed_valid_fmt) if per_seed_valid_fmt else None,
            }

    # Arm-level summary (avg across tasks)
    for arm in arms:
        beams, samples = [], []
        for task in SID_TASKS:
            entry = out["per_task"][task].get(arm, {})
            if entry.get("beam_catalog_hit_mean") is not None:
                beams.append(entry["beam_catalog_hit_mean"])
            if entry.get("sample_any_catalog_hit_mean") is not None:
                samples.append(entry["sample_any_catalog_hit_mean"])
        out["per_arm"][arm] = {
            "avg_beam_catalog_hit": statistics.mean(beams) if beams else None,
            "avg_sample_any_catalog_hit": statistics.mean(samples) if samples else None,
            "n_tasks": len(beams),
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print("\nper_arm summary (avg across 8 SID tasks):")
    for arm, v in out["per_arm"].items():
        b = v["avg_beam_catalog_hit"]
        s = v["avg_sample_any_catalog_hit"]
        print(f"  {arm}: beam-level hit = {b:.4f}" + (f", any-beam sample hit = {s:.4f}" if s is not None else "") + f"  (tasks={v['n_tasks']})")


if __name__ == "__main__":
    main()
