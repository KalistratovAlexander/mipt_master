#!/usr/bin/env python3
"""H1 Section F: McNemar (F1) + paired bootstrap CI (F2) on Δ = 8B − 1.7B.

Reads two `eval_unified_*.json` files (each must contain inline metrics from
the patched evaluator) and dumps `h1_stat_tests.json` with:
- mcnemar per task on hit@10 (m=8, Bonferroni α=0.05/8 ≈ 0.00625)
- paired bootstrap CI (95% and Bonferroni-corrected) per (task, metric)

Bootstrap m count = 81 (8 SID tasks × 6 prompt metrics + 8 head + 8 tail
+ 8 coverage@10 + 3 text tasks × 3 metrics). valid_format (saturated near
1.0) excluded — no useful signal in CI.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl

SID_PATTERN = re.compile(r"<\|A(\d+)\|><\|B(\d+)\|><\|C(\d+)\|><\|D(\d+)\|>")

SID_TASKS = [
    "title_to_sid", "description_to_sid", "features_to_sid",
    "copurchase_forward", "copurchase_backward",
    "seq_last_2", "seq_last_3", "seq_last_5",
]
TEXT_TASKS = ["sid_to_title", "sid_to_description", "sid_to_features"]
TEXT_METRICS = ["char_f1", "rouge_l", "cosine_sim"]
SID_PROMPT_METRICS = [
    "hit@10", "ndcg@10",
    "hier_hit@10.A", "hier_hit@10.AB", "hier_hit@10.ABC",
    "catalog_hit_top1",
]


# ---------------------------------------------------------------------------
# Helpers (duplicated from evaluate_unified.py to avoid torch import drag)
# ---------------------------------------------------------------------------

def parse_sid(text: str) -> Optional[Tuple[str, str, str, str]]:
    m = SID_PATTERN.search(text)
    return tuple(m.groups()) if m else None


def hierarchical_match_depth(pred, gold) -> int:
    for i in range(4):
        if str(pred[i]) != str(gold[i]):
            return i
    return 4


def build_catalog_sids_tuples(sid_file: Path) -> Set[Tuple[str, str, str, str]]:
    df = pl.read_parquet(str(sid_file))
    out: Set[Tuple[str, str, str, str]] = set()
    for r in df.iter_rows(named=True):
        a, b, c, d = r.get("A"), r.get("B"), r.get("C"), r.get("D")
        if a is not None and b is not None and c is not None and d is not None:
            out.add((str(int(a)), str(int(b)), str(int(c)), str(int(d))))
    return out


def build_head_tail_sids(
    train_seq_file: Path, pct: float = 0.20,
) -> Tuple[Set[Tuple[str, str, str, str]], Set[Tuple[str, str, str, str]], int]:
    df = pl.read_parquet(str(train_seq_file))
    counts: Counter = Counter()
    for sid_seq in df["sid_sequence"].to_list():
        if not sid_seq:
            continue
        for s in sid_seq:
            p = parse_sid(s)
            if p is not None:
                counts[p] += 1
    if not counts:
        return set(), set(), 0
    n = len(counts)
    k = max(int(n * pct), 1)
    sorted_desc = sorted(counts.keys(), key=lambda s: -counts[s])
    return set(sorted_desc[:k]), set(sorted_desc[-k:]), n


# ---------------------------------------------------------------------------
# Per-prompt metric derivation (paired between models)
# ---------------------------------------------------------------------------

def derive_sid_per_prompt(
    beam_data: dict,
    catalog_tuples: Set[Tuple[str, str, str, str]],
    head_sids: Set[Tuple[str, str, str, str]],
    tail_sids: Set[Tuple[str, str, str, str]],
) -> Dict[str, list]:
    psg = beam_data.get("per_sample_gold", [])
    psbp = beam_data.get("per_sample_beam_preds", [])
    psh10 = beam_data.get("per_sample_hit@10", [])
    n = len(psg)
    out: Dict[str, list] = {m: [] for m in SID_PROMPT_METRICS}
    head_idx: List[int] = []
    tail_idx: List[int] = []
    for i in range(n):
        g = psg[i]
        bp = psbp[i] if i < len(psbp) else []
        out["hit@10"].append(float(psh10[i]) if i < len(psh10) else 0.0)
        ndcg = 0.0
        if g is not None:
            for rank, p in enumerate(bp[:10], 1):
                if p is not None and tuple(p) == tuple(g):
                    ndcg = 1.0 / np.log2(rank + 1)
                    break
        out["ndcg@10"].append(ndcg)
        for level, mind in [("A", 1), ("AB", 2), ("ABC", 3)]:
            h = 0.0
            if g is not None:
                for p in bp[:10]:
                    if p is not None and hierarchical_match_depth(p, g) >= mind:
                        h = 1.0
                        break
            out[f"hier_hit@10.{level}"].append(h)
        ch = 0.0
        if bp and bp[0] is not None and tuple(bp[0]) in catalog_tuples:
            ch = 1.0
        out["catalog_hit_top1"].append(ch)
        if g is not None:
            gt = tuple(g)
            if gt in head_sids:
                head_idx.append(i)
            elif gt in tail_sids:
                tail_idx.append(i)
    out["_head_idx"] = head_idx
    out["_tail_idx"] = tail_idx
    return out


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def mcnemar_test(arr_a, arr_b) -> dict:
    from scipy.stats import binom, chi2  # lazy: scipy missing in some local envs
    a = np.asarray([int(x) for x in arr_a])
    b_arr = np.asarray([int(x) for x in arr_b])
    b = int(((a == 0) & (b_arr == 1)).sum())
    c = int(((a == 1) & (b_arr == 0)).sum())
    n_disc = b + c
    n = int(len(a))
    if n_disc == 0:
        return {"p_value": 1.0, "b": 0, "c": 0, "method": "no_discordant", "n": n}
    if min(b, c) < 25:
        x = min(b, c)
        p = min(2 * float(binom.cdf(x, n_disc, 0.5)), 1.0)
        method = "exact"
    else:
        stat = (abs(b - c) - 1) ** 2 / n_disc
        p = float(chi2.sf(stat, 1))
        method = "chi2_yates"
    return {"p_value": p, "b": b, "c": c, "method": method, "n": n}


def paired_bootstrap_deltas(arr_a, arr_b, n_iter: int, rng: np.random.Generator) -> np.ndarray:
    a = np.asarray(arr_a, dtype=float)
    b = np.asarray(arr_b, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return np.zeros(n_iter)
    diff = b[:n] - a[:n]
    idx = rng.integers(0, n, size=(n_iter, n))
    return diff[idx].mean(axis=1)


def ci_from_deltas(deltas: np.ndarray, alpha: float) -> dict:
    if len(deltas) == 0:
        return {"ci_low": 0.0, "ci_high": 0.0, "significant": False}
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool((0 < lo) or (hi < 0)),
    }


def bootstrap_block(
    arr_a, arr_b, n_iter: int, alpha: float, alpha_bonf: float, rng: np.random.Generator,
) -> dict:
    a = np.asarray(arr_a, dtype=float)
    b = np.asarray(arr_b, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return {
            "delta": 0.0, "n": 0,
            "ci95": {"ci_low": 0.0, "ci_high": 0.0, "significant": False},
            "ci_bonf": {"ci_low": 0.0, "ci_high": 0.0, "significant": False},
        }
    deltas = paired_bootstrap_deltas(a, b, n_iter, rng)
    return {
        "delta": float((b[:n] - a[:n]).mean()),
        "n": int(n),
        "ci95": ci_from_deltas(deltas, alpha),
        "ci_bonf": ci_from_deltas(deltas, alpha_bonf),
    }


def coverage_bootstrap(
    beam_preds_a, beam_preds_b, catalog_size: int,
    n_iter: int, alpha: float, alpha_bonf: float, rng: np.random.Generator,
) -> dict:
    n = min(len(beam_preds_a), len(beam_preds_b))
    if n == 0 or catalog_size == 0:
        return {
            "delta": 0.0, "n": 0,
            "ci95": {"ci_low": 0.0, "ci_high": 0.0, "significant": False},
            "ci_bonf": {"ci_low": 0.0, "ci_high": 0.0, "significant": False},
        }
    sets_a = [frozenset(tuple(s) for s in bms if s is not None) for bms in beam_preds_a[:n]]
    sets_b = [frozenset(tuple(s) for s in bms if s is not None) for bms in beam_preds_b[:n]]
    deltas = np.empty(n_iter)
    for it in range(n_iter):
        idx = rng.integers(0, n, size=n)
        ua: Set = set()
        ub: Set = set()
        for i in idx:
            ua |= sets_a[i]
            ub |= sets_b[i]
        deltas[it] = (len(ub) - len(ua)) / catalog_size
    full_a = len(set().union(*sets_a)) / catalog_size if sets_a else 0.0
    full_b = len(set().union(*sets_b)) / catalog_size if sets_b else 0.0
    return {
        "delta": float(full_b - full_a),
        "n": int(n),
        "ci95": ci_from_deltas(deltas, alpha),
        "ci_bonf": ci_from_deltas(deltas, alpha_bonf),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-1p7b", required=True, help="JSON from evaluate_unified.py for 1.7B")
    ap.add_argument("--eval-8b", required=True, help="JSON from evaluate_unified.py for 8B")
    ap.add_argument("--catalog", required=True,
                    help="Pet_Supplies_items_with_semantic_ids.parquet")
    ap.add_argument("--train-sequences", required=True,
                    help="Pet_Supplies_sequences_with_sid_train.parquet")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--n-bootstrap-coverage", type=int, default=2000,
                    help="Coverage bootstrap is O(n_iter * n_prompts * union_ops); use fewer iters.")
    ap.add_argument("--head-tail-pct", type=float, default=0.20)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.eval_1p7b}", file=sys.stderr)
    res_a = json.load(open(args.eval_1p7b))
    print(f"Loading {args.eval_8b}", file=sys.stderr)
    res_b = json.load(open(args.eval_8b))

    print(f"Building catalog from {args.catalog}", file=sys.stderr)
    catalog = build_catalog_sids_tuples(Path(args.catalog))
    print(f"  catalog SIDs: {len(catalog):,}", file=sys.stderr)

    print(f"Building head/tail from {args.train_sequences}", file=sys.stderr)
    head_sids, tail_sids, n_unique = build_head_tail_sids(
        Path(args.train_sequences), pct=args.head_tail_pct
    )
    print(f"  head: {len(head_sids):,} / tail: {len(tail_sids):,} (of {n_unique:,} train SIDs)",
          file=sys.stderr)

    n_mcnemar = len(SID_TASKS)  # 8
    n_bootstrap_metrics = (
        len(SID_TASKS) * len(SID_PROMPT_METRICS)         # 8*6 = 48
        + len(SID_TASKS) * 2                              # head + tail = 16
        + len(SID_TASKS)                                  # coverage@10 = 8
        + len(TEXT_TASKS) * len(TEXT_METRICS)             # 9
    )  # = 81

    alpha_mc_bonf = args.alpha / n_mcnemar
    alpha_bs_bonf = args.alpha / n_bootstrap_metrics

    out: Dict = {
        "meta": {
            "n_bootstrap": args.n_bootstrap,
            "n_bootstrap_coverage": args.n_bootstrap_coverage,
            "alpha": args.alpha,
            "head_tail_pct": args.head_tail_pct,
            "seed": args.seed,
            "m_mcnemar": n_mcnemar,
            "m_bootstrap": n_bootstrap_metrics,
            "alpha_mcnemar_bonf": alpha_mc_bonf,
            "alpha_bootstrap_bonf": alpha_bs_bonf,
            "eval_1p7b": str(args.eval_1p7b),
            "eval_8b": str(args.eval_8b),
            "catalog_size": len(catalog),
            "n_unique_train_sids": n_unique,
            "head_size": len(head_sids),
            "tail_size": len(tail_sids),
        },
        "mcnemar": {},
        "bootstrap_ci": {},
    }

    for t in SID_TASKS:
        if t not in res_a.get("tasks", {}) or t not in res_b.get("tasks", {}):
            print(f"  warn: task {t} missing in one of the JSONs", file=sys.stderr)
            continue
        beam_a = res_a["tasks"][t].get("beam", {})
        beam_b = res_b["tasks"][t].get("beam", {})
        if not beam_a.get("per_sample_gold") or not beam_b.get("per_sample_gold"):
            print(f"  warn: task {t} missing per_sample_* arrays — re-run evaluator", file=sys.stderr)
            continue
        per_a = derive_sid_per_prompt(beam_a, catalog, head_sids, tail_sids)
        per_b = derive_sid_per_prompt(beam_b, catalog, head_sids, tail_sids)

        mc = mcnemar_test(per_a["hit@10"], per_b["hit@10"])
        mc["significant_bonf"] = bool(mc["p_value"] < alpha_mc_bonf)
        out["mcnemar"][t] = mc

        out["bootstrap_ci"][t] = {}
        for m in SID_PROMPT_METRICS:
            out["bootstrap_ci"][t][m] = bootstrap_block(
                per_a[m], per_b[m], args.n_bootstrap, args.alpha, alpha_bs_bonf, rng,
            )

        head_common = sorted(set(per_a["_head_idx"]) & set(per_b["_head_idx"]))
        tail_common = sorted(set(per_a["_tail_idx"]) & set(per_b["_tail_idx"]))
        if head_common:
            out["bootstrap_ci"][t]["head_recall@10"] = bootstrap_block(
                [per_a["hit@10"][i] for i in head_common],
                [per_b["hit@10"][i] for i in head_common],
                args.n_bootstrap, args.alpha, alpha_bs_bonf, rng,
            )
        if tail_common:
            out["bootstrap_ci"][t]["tail_recall@10"] = bootstrap_block(
                [per_a["hit@10"][i] for i in tail_common],
                [per_b["hit@10"][i] for i in tail_common],
                args.n_bootstrap, args.alpha, alpha_bs_bonf, rng,
            )

        print(f"  coverage bootstrap for {t}...", file=sys.stderr)
        out["bootstrap_ci"][t]["coverage@10"] = coverage_bootstrap(
            beam_a.get("per_sample_beam_preds", []),
            beam_b.get("per_sample_beam_preds", []),
            len(catalog),
            args.n_bootstrap_coverage, args.alpha, alpha_bs_bonf, rng,
        )

    for t in TEXT_TASKS:
        if t not in res_a.get("tasks", {}) or t not in res_b.get("tasks", {}):
            print(f"  warn: text task {t} missing", file=sys.stderr)
            continue
        text_a = res_a["tasks"][t].get("text_metrics", {}).get("per_sample", {})
        text_b = res_b["tasks"][t].get("text_metrics", {}).get("per_sample", {})
        if not text_a or not text_b:
            print(f"  warn: text task {t} missing per_sample — re-run with patched evaluator",
                  file=sys.stderr)
            continue
        out["bootstrap_ci"][t] = {}
        for m in TEXT_METRICS:
            arr_a = text_a.get(m, [])
            arr_b = text_b.get(m, [])
            if not arr_a or not arr_b:
                continue
            out["bootstrap_ci"][t][m] = bootstrap_block(
                arr_a, arr_b, args.n_bootstrap, args.alpha, alpha_bs_bonf, rng,
            )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
