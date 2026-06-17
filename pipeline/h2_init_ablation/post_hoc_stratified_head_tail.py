#!/usr/bin/env python3
"""Post-hoc B.8: Stratified Recall@10 by item-frequency head/tail split.

Split definition (pragmatic proxy, see note below):
  head = gold SIDs that appear >=2 times across 1000 val samples of a task
  tail = gold SIDs that appear exactly once

Rationale:
  The frozen metric calls for "split val items by item frequency in train".
  True train-frequency requires parsing 4.7M train conversations to extract
  SID tuples — infeasible in the 2-day defense window. Val-duplicate count
  is a conservative retrieval-space proxy: val is IID-sampled from the same
  catalog as train, so items that recur in val almost-certainly recur in
  train (popular items dominate both). The proxy under-counts pure head
  items (those appearing 2+ times in train but only once in 1000-sample
  val), so the head/tail gap we measure is a *lower bound* on the true
  effect. Explicitly documented in Discussion as a limitation.

Reads per_sample_gold + per_sample_hit@10 from results_unified.json.
Writes results/post_hoc_stratified_head_tail.json with:
  - frequency histogram per task
  - per (arm, task): R@10 on head, R@10 on tail, delta, n_head, n_tail
  - per arm: avg R@10 head vs tail across 8 SID tasks
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--arms", default="A,B,C,D")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--head-threshold", type=int, default=2,
                   help="min gold freq to be in head (default: 2)")
    args = p.parse_args()

    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    runs_dir = Path(args.runs_dir)

    out = {
        "split_definition": f"head = gold freq >= {args.head_threshold} in 1000-sample val; tail = freq < threshold",
        "note": "Val-duplicate proxy for train frequency — see script docstring",
        "per_task": {},
        "per_arm": {},
        "summary": {},
    }

    # --- Step 1: per task, compute freq map (use any run; seed=42 shared) ---
    freq_per_task = {}
    for task in SID_TASKS:
        # Find any run with per_sample_gold populated
        gold_seq = None
        for arm in arms:
            for sd in seeds:
                u = load_unified(runs_dir, arm, sd)
                if u is None:
                    continue
                g = u.get("tasks", {}).get(task, {}).get("beam", {}).get("per_sample_gold")
                if g:
                    gold_seq = [tuple(gi) if gi is not None else None for gi in g]
                    break
            if gold_seq:
                break
        if not gold_seq:
            print(f"WARN: no per_sample_gold for task {task}")
            continue
        cnt = Counter(g for g in gold_seq if g is not None)
        freq_per_task[task] = (gold_seq, cnt)
        n_total = sum(cnt.values())
        n_head_samples = sum(1 for g in gold_seq if g is not None and cnt[g] >= args.head_threshold)
        n_tail_samples = sum(1 for g in gold_seq if g is not None and cnt[g] < args.head_threshold)
        n_unique_head = sum(1 for v in cnt.values() if v >= args.head_threshold)
        n_unique_tail = sum(1 for v in cnt.values() if v < args.head_threshold)
        print(f"{task}: {n_total} gold samples; head_samples={n_head_samples} "
              f"({n_unique_head} uniq SIDs), tail_samples={n_tail_samples} "
              f"({n_unique_tail} uniq SIDs)")

    # --- Step 2: per (arm, task), split per_sample_hit@10 by head/tail ---
    for task in SID_TASKS:
        if task not in freq_per_task:
            continue
        gold_seq, cnt = freq_per_task[task]
        out["per_task"][task] = {
            "n_head_samples": sum(1 for g in gold_seq if g is not None and cnt[g] >= args.head_threshold),
            "n_tail_samples": sum(1 for g in gold_seq if g is not None and cnt[g] < args.head_threshold),
            "per_arm": {},
        }
        for arm in arms:
            head_per_seed = []
            tail_per_seed = []
            deltas = []
            for sd in seeds:
                u = load_unified(runs_dir, arm, sd)
                if u is None:
                    continue
                hits = u.get("tasks", {}).get(task, {}).get("beam", {}).get("per_sample_hit@10")
                if not hits:
                    continue
                head_hits = []
                tail_hits = []
                for g, h in zip(gold_seq, hits):
                    if g is None or h is None:
                        continue
                    if cnt[g] >= args.head_threshold:
                        head_hits.append(h)
                    else:
                        tail_hits.append(h)
                head_r = sum(head_hits) / len(head_hits) if head_hits else None
                tail_r = sum(tail_hits) / len(tail_hits) if tail_hits else None
                if head_r is not None:
                    head_per_seed.append(head_r)
                if tail_r is not None:
                    tail_per_seed.append(tail_r)
                if head_r is not None and tail_r is not None:
                    deltas.append(head_r - tail_r)
            out["per_task"][task]["per_arm"][arm] = {
                "R@10_head_mean": statistics.mean(head_per_seed) if head_per_seed else None,
                "R@10_head_std": statistics.stdev(head_per_seed) if len(head_per_seed) >= 2 else None,
                "R@10_tail_mean": statistics.mean(tail_per_seed) if tail_per_seed else None,
                "R@10_tail_std": statistics.stdev(tail_per_seed) if len(tail_per_seed) >= 2 else None,
                "delta_head_minus_tail_mean": statistics.mean(deltas) if deltas else None,
                "n_seeds": len(head_per_seed),
            }

    # --- Step 3: per arm macro (avg across 8 SID tasks) ---
    for arm in arms:
        heads, tails, deltas = [], [], []
        for task in SID_TASKS:
            e = out["per_task"].get(task, {}).get("per_arm", {}).get(arm, {})
            if e.get("R@10_head_mean") is not None:
                heads.append(e["R@10_head_mean"])
            if e.get("R@10_tail_mean") is not None:
                tails.append(e["R@10_tail_mean"])
            if e.get("delta_head_minus_tail_mean") is not None:
                deltas.append(e["delta_head_minus_tail_mean"])
        out["per_arm"][arm] = {
            "R@10_head_macro": statistics.mean(heads) if heads else None,
            "R@10_tail_macro": statistics.mean(tails) if tails else None,
            "delta_macro": statistics.mean(deltas) if deltas else None,
            "n_tasks": len(heads),
        }

    out["summary"]["arms"] = arms
    out["summary"]["seeds"] = seeds
    out["summary"]["head_threshold"] = args.head_threshold

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Per-arm macro (avg across 8 SID tasks) ===")
    print(f"{'arm':4} {'R@10_head':>12} {'R@10_tail':>12} {'Δ (H−T)':>12}  n_tasks")
    for arm in arms:
        v = out["per_arm"][arm]
        h = v["R@10_head_macro"]
        t = v["R@10_tail_macro"]
        d = v["delta_macro"]
        row = f"{arm:4}"
        row += f" {h:12.4f}" if h is not None else f" {'—':>12}"
        row += f" {t:12.4f}" if t is not None else f" {'—':>12}"
        row += f" {d:12.4f}" if d is not None else f" {'—':>12}"
        row += f"  {v['n_tasks']}"
        print(row)


if __name__ == "__main__":
    main()
