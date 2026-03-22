#!/usr/bin/env python3
"""Unified evaluation of Semantic ID models for master's thesis.

Produces identical JSON output for any model (1.8B, 8B, etc.) enabling
direct comparison across all metrics.

Metrics follow TIGER (arXiv:2305.05065), PLUM (arXiv:2510.07784),
and OpenOneRec (arXiv:2512.24762).

SID Prediction metrics:
  - Greedy: exact_match, level_A/AB/ABC/ABCD, format compliance
  - Beam:   Hit@1/5/10, MRR@10, NDCG@10, hierarchical variants
  - Hallucination rate, diversity, intent match, error analysis

Text Generation metrics (SID→Text):
  - ROUGE-L, Token F1, Token Jaccard, Char F1
  - Cosine similarity (sentence-transformers, optional)
  - Category: level_match@1-4, exact_ref

Performance:
  - TTFT (Time to First Token, prefill latency)
  - TPS (Tokens Per Second, SID + text generation)
  - E2E latency (greedy SID, greedy text, beam search)
  - GPU memory (weights, peak)
  - GPU power draw, energy per request

General:
  - Perplexity on WikiText-2 (catastrophic forgetting)

Usage:
  python pipeline/evaluate_unified.py \
    --model-path vast_8b/stage2_full_finetune/output/final \
    --data-dir data \
    --model-name "8B" \
    --samples-per-task 200 \
    --beam-size 10 \
    --output results/eval_8b.json

  python pipeline/evaluate_unified.py \
    --model-path vast/stage2_full_finetune/output/stage2_h100/final \
    --data-dir data \
    --model-name "1.8B" \
    --samples-per-task 200 \
    --output results/eval_1.8b.json
"""

import argparse
import json
import logging
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

_THINK_BLOCK = "<think>\n\n</think>\n\n"

SID_PATTERN = re.compile(
    r"<\|sid_start\|>"
    r"<\|A(\d+)\|><\|B(\d+)\|><\|C(\d+)\|><\|D(\d+)\|>"
    r"<\|sid_end\|>"
)
SID_PATTERN_FULL = re.compile(r"<\|sid_start\|>(<\|[ABCD]\d+\|>)+<\|sid_end\|>")

# ---- Tier 1: Core SID prediction (1000 samples, greedy + beam) ----
SID_PREDICTION_TASKS = {
    "copurchase_forward", "copurchase_backward",
    "seq_last_2", "seq_last_3", "seq_last_5",
    "title_to_sid", "description_to_sid", "features_to_sid",
}

# ---- Tier 2: SID→Text decoding (1000 samples, greedy only) ----
TEXT_GENERATION_TASKS = {
    "sid_to_title", "sid_to_description", "sid_to_features",
}

ALL_KNOWN_TASKS = SID_PREDICTION_TASKS | TEXT_GENERATION_TASKS

# Per-tier default sample counts (overridden by --samples-per-task or --dry-run)
TIER_SAMPLES = {
    "sid_prediction": 1000,
    "text_generation": 1000,
}

TASK_GROUPS = {
    "collaborative_signal": ["copurchase_forward", "copurchase_backward",
                             "seq_last_2", "seq_last_3", "seq_last_5"],
    "text_to_sid": ["title_to_sid", "description_to_sid", "features_to_sid"],
    "sid_to_text": ["sid_to_title", "sid_to_description", "sid_to_features"],
}

TASK_TO_CATALOG_FIELD = {
    "sid_to_title": "title",
    "sid_to_description": "description_text",
    "sid_to_features": "features_text",
}

CATEGORY_KEYWORDS = {
    "dog": "dog", "dogs": "dog", "puppy": "dog", "puppies": "dog", "canine": "dog",
    "cat": "cat", "cats": "cat", "kitten": "cat", "kittens": "cat", "feline": "cat",
    "fish": "fish", "aquarium": "fish", "tank": "fish", "marine": "fish",
    "bird": "bird", "birds": "bird", "parrot": "bird", "parakeet": "bird",
    "hamster": "small_animal", "rabbit": "small_animal", "guinea pig": "small_animal",
    "rat": "small_animal", "mouse": "small_animal", "gerbil": "small_animal",
    "reptile": "reptile", "snake": "reptile", "lizard": "reptile", "turtle": "reptile",
}


# ============================================================================
# SID Parsing
# ============================================================================

def parse_sid(text: str) -> Optional[Tuple[str, str, str, str]]:
    """Parse SID from text, return (A, B, C, D) as strings or None."""
    m = SID_PATTERN.search(text)
    return tuple(m.groups()) if m else None


def extract_sid_token(text: str) -> Optional[str]:
    """Extract full SID token string like <|sid_start|>...<|sid_end|>."""
    m = SID_PATTERN_FULL.search(text)
    return m.group(0) if m else None


def sid_tuple_to_token(sid: Tuple[str, str, str, str]) -> str:
    return f"<|sid_start|><|A{sid[0]}|><|B{sid[1]}|><|C{sid[2]}|><|D{sid[3]}|><|sid_end|>"


def parse_all_sids(text: str) -> List[Tuple[str, str, str, str]]:
    """Extract ALL SIDs from text (for multi-SID gold answers)."""
    return [tuple(m.groups()) for m in SID_PATTERN.finditer(text)]


def hierarchical_match_depth(pred: Tuple, gold: Tuple) -> int:
    """Return match depth: 0=none, 1=A, 2=AB, 3=ABC, 4=ABCD (exact)."""
    for i in range(4):
        if pred[i] != gold[i]:
            return i
    return 4


# ============================================================================
# Bootstrap CI
# ============================================================================

def bootstrap_ci(
    values: list,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high) via percentile bootstrap."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(rng.choices(values, k=n)) / n
        for _ in range(n_bootstrap)
    )
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_bootstrap)]
    hi = means[int((1 - alpha) * n_bootstrap)]
    return sum(values) / n, lo, hi


# ============================================================================
# Text Metrics (NLG)
# ============================================================================

def _tokenize_normalize(text: str) -> List[str]:
    text = text.lower().strip()
    text = re.sub(r"[\"()\[\]{}<>]", " ", text)
    text = re.sub(r"([.,!?;:])\s", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return [t for t in text.split() if t]


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower().strip())


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize_normalize(prediction)
    ref_tokens = _tokenize_normalize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec = lcs / n
    rec = lcs / m
    return 2 * prec * rec / (prec + rec)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(_tokenize_normalize(prediction))
    ref_tokens = set(_tokenize_normalize(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(ref_tokens)
    return 2 * prec * rec / (prec + rec)


def token_jaccard(a: str, b: str) -> float:
    ta = set(_tokenize_words(a))
    tb = set(_tokenize_words(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def char_f1(a: str, b: str) -> float:
    a_clean = re.sub(r"\s+", "", a.lower().strip())
    b_clean = re.sub(r"\s+", "", b.lower().strip())
    if not a_clean and not b_clean:
        return 1.0
    if not a_clean or not b_clean:
        return 0.0
    ca = Counter(a_clean)
    cb = Counter(b_clean)
    common = sum((ca & cb).values())
    prec = common / max(1, sum(ca.values()))
    rec = common / max(1, sum(cb.values()))
    return 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))


# ============================================================================
# Category Metrics
# ============================================================================

def split_category_path(cat: str) -> List[str]:
    cat = (cat or "").strip()
    if not cat:
        return []
    return [p.strip() for p in cat.split(">") if p.strip()]


def category_prefix_depth(gold: str, pred: str) -> int:
    g = split_category_path(gold)
    p = split_category_path(pred)
    depth = 0
    for gi, pi in zip(g, p):
        if gi.lower().strip() == pi.lower().strip():
            depth += 1
        else:
            break
    return depth


def category_level_match(gold: str, pred: str, level: int) -> bool:
    g = split_category_path(gold)
    p = split_category_path(pred)
    if len(g) < level or len(p) < level:
        return False
    return all(
        g[i].lower().strip() == p[i].lower().strip()
        for i in range(level)
    )


# ============================================================================
# Similarity Backend (optional sentence-transformers)
# ============================================================================

class SimilarityBackend:
    def __init__(self, device: str = "cpu", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = device
        self.model_name = model_name
        self.backend = None
        self._st = None
        self._init_backend()

    def _init_backend(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.model_name, device=self.device)
            self.backend = "sentence_transformers"
            log.info(f"Similarity backend: sentence-transformers ({self.model_name})")
            return
        except Exception:
            pass
        try:
            from sklearn.feature_extraction.text import HashingVectorizer
            self._hashing = HashingVectorizer(
                n_features=2**18, alternate_sign=False, norm="l2",
                ngram_range=(1, 2), lowercase=True,
            )
            self.backend = "hashing"
            log.info("Similarity backend: hashing (fallback)")
            return
        except Exception:
            self.backend = None
            log.warning("No similarity backend available")

    def available(self) -> bool:
        return self.backend is not None

    def name(self) -> str:
        if self.backend == "sentence_transformers":
            return f"embedding_cosine(st:{self.model_name})"
        if self.backend == "hashing":
            return "cosine(hashing_1-2gram)"
        return "none"

    def cosine_pairs(self, texts_a: List[str], texts_b: List[str]) -> List[float]:
        if not self.available() or not texts_a:
            return [0.0] * len(texts_a)
        if self.backend == "sentence_transformers":
            emba = self._st.encode(texts_a, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
            embb = self._st.encode(texts_b, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
            return (np.sum(emba * embb, axis=1)).astype(float).tolist()
        if self.backend == "hashing":
            Xa = self._hashing.transform(texts_a)
            Xb = self._hashing.transform(texts_b)
            sims = np.asarray((Xa.multiply(Xb)).sum(axis=1)).reshape(-1)
            return sims.astype(float).tolist()
        return [0.0] * len(texts_a)


# ============================================================================
# SID Greedy Metrics
# ============================================================================

@dataclass
class SIDGreedyMetrics:
    total: int = 0
    valid_format: int = 0
    level_a: int = 0
    level_ab: int = 0
    level_abc: int = 0
    level_abcd: int = 0
    exact: int = 0
    per_sample_exact: list = field(default_factory=list)
    per_sample_a: list = field(default_factory=list)

    def update(self, pred: Optional[Tuple], gold: Optional[Tuple]):
        self.total += 1
        if pred is None or gold is None:
            self.per_sample_exact.append(0.0)
            self.per_sample_a.append(0.0)
            return
        self.valid_format += 1
        depth = hierarchical_match_depth(pred, gold)
        self.per_sample_a.append(float(depth >= 1))
        self.per_sample_exact.append(float(depth == 4))
        if depth >= 1: self.level_a += 1
        if depth >= 2: self.level_ab += 1
        if depth >= 3: self.level_abc += 1
        if depth >= 4:
            self.level_abcd += 1
            self.exact += 1

    def to_dict(self) -> dict:
        n = max(self.total, 1)
        exact_mean, exact_lo, exact_hi = bootstrap_ci(self.per_sample_exact)
        a_mean, a_lo, a_hi = bootstrap_ci(self.per_sample_a)
        return {
            "n": self.total,
            "valid_format": round(self.valid_format / n, 4),
            "level_A": round(self.level_a / n, 4),
            "level_A_ci95": [round(a_lo, 4), round(a_hi, 4)],
            "level_AB": round(self.level_ab / n, 4),
            "level_ABC": round(self.level_abc / n, 4),
            "level_ABCD": round(self.level_abcd / n, 4),
            "exact_match": round(self.exact / n, 4),
            "exact_match_ci95": [round(exact_lo, 4), round(exact_hi, 4)],
        }


# ============================================================================
# SID Beam Metrics
# ============================================================================

@dataclass
class SIDBeamMetrics:
    """Hit@K, MRR@K, NDCG@K — standard RecSys metrics from beam search."""
    k: int = 10
    total: int = 0
    per_sample_hit1: list = field(default_factory=list)
    per_sample_hit5: list = field(default_factory=list)
    per_sample_hit10: list = field(default_factory=list)
    per_sample_rr: list = field(default_factory=list)
    per_sample_ndcg: list = field(default_factory=list)
    # Hierarchical: hit@K at each level
    hier_hit5: dict = field(default_factory=lambda: {"A": [], "AB": [], "ABC": [], "ABCD": []})
    hier_hit10: dict = field(default_factory=lambda: {"A": [], "AB": [], "ABC": [], "ABCD": []})
    hier_mrr10: dict = field(default_factory=lambda: {"A": [], "AB": [], "ABC": [], "ABCD": []})

    def update(self, beam_preds: List[Optional[Tuple]], gold: Optional[Tuple]):
        self.total += 1
        if gold is None:
            for lst in [self.per_sample_hit1, self.per_sample_hit5, self.per_sample_hit10,
                        self.per_sample_rr, self.per_sample_ndcg]:
                lst.append(0.0)
            for level in ["A", "AB", "ABC", "ABCD"]:
                self.hier_hit5[level].append(0.0)
                self.hier_hit10[level].append(0.0)
                self.hier_mrr10[level].append(0.0)
            return

        # Exact match metrics
        hit1 = float(len(beam_preds) > 0 and beam_preds[0] == gold)
        hit5 = float(gold in beam_preds[:5])
        hit10 = float(gold in beam_preds[:10])
        self.per_sample_hit1.append(hit1)
        self.per_sample_hit5.append(hit5)
        self.per_sample_hit10.append(hit10)

        # MRR
        rr = 0.0
        for rank, pred in enumerate(beam_preds[:self.k], 1):
            if pred == gold:
                rr = 1.0 / rank
                break
        self.per_sample_rr.append(rr)

        # NDCG@K (single relevant item)
        ndcg = 0.0
        for rank, pred in enumerate(beam_preds[:self.k], 1):
            if pred == gold:
                ndcg = 1.0 / math.log2(rank + 1)
                break
        self.per_sample_ndcg.append(ndcg)

        # Hierarchical metrics
        level_thresholds = {"A": 1, "AB": 2, "ABC": 3, "ABCD": 4}
        for level_name, min_depth in level_thresholds.items():
            # hier hit@5
            h5 = 0.0
            for p in beam_preds[:5]:
                if p is not None and hierarchical_match_depth(p, gold) >= min_depth:
                    h5 = 1.0
                    break
            self.hier_hit5[level_name].append(h5)

            # hier hit@10
            h10 = 0.0
            for p in beam_preds[:10]:
                if p is not None and hierarchical_match_depth(p, gold) >= min_depth:
                    h10 = 1.0
                    break
            self.hier_hit10[level_name].append(h10)

            # hier mrr@10
            hmrr = 0.0
            for rank, p in enumerate(beam_preds[:self.k], 1):
                if p is not None and hierarchical_match_depth(p, gold) >= min_depth:
                    hmrr = 1.0 / rank
                    break
            self.hier_mrr10[level_name].append(hmrr)

    def to_dict(self) -> dict:
        n = max(self.total, 1)
        h1_m, h1_lo, h1_hi = bootstrap_ci(self.per_sample_hit1)
        h5_m, h5_lo, h5_hi = bootstrap_ci(self.per_sample_hit5)
        h10_m, h10_lo, h10_hi = bootstrap_ci(self.per_sample_hit10)
        mrr_m, mrr_lo, mrr_hi = bootstrap_ci(self.per_sample_rr)
        ndcg_m, ndcg_lo, ndcg_hi = bootstrap_ci(self.per_sample_ndcg)

        result = {
            "n": self.total,
            "hit@1": round(h1_m, 4),
            "hit@1_ci95": [round(h1_lo, 4), round(h1_hi, 4)],
            "hit@5": round(h5_m, 4),
            "hit@5_ci95": [round(h5_lo, 4), round(h5_hi, 4)],
            "hit@10": round(h10_m, 4),
            "hit@10_ci95": [round(h10_lo, 4), round(h10_hi, 4)],
            "mrr@10": round(mrr_m, 4),
            "mrr@10_ci95": [round(mrr_lo, 4), round(mrr_hi, 4)],
            "ndcg@10": round(ndcg_m, 4),
            "ndcg@10_ci95": [round(ndcg_lo, 4), round(ndcg_hi, 4)],
        }

        # Hierarchical
        for metric_name, data in [("hier_hit@5", self.hier_hit5),
                                   ("hier_hit@10", self.hier_hit10),
                                   ("hier_mrr@10", self.hier_mrr10)]:
            sub = {}
            for level_name, values in data.items():
                m, lo, hi = bootstrap_ci(values)
                sub[level_name] = round(m, 4)
            result[metric_name] = sub

        return result


# ============================================================================
# Multi-SID Metrics (for tasks with multiple gold SIDs)
# ============================================================================

@dataclass
class MultiSIDGreedyMetrics:
    """Greedy metrics when gold contains multiple SIDs.

    exact_match = predicted SID is in the gold set
    level_X     = predicted SID matches any gold SID at level X
    coverage    = fraction of gold SIDs matched by beam candidates
    """
    total: int = 0
    valid_format: int = 0
    exact: int = 0
    level_a: int = 0
    level_ab: int = 0
    level_abc: int = 0
    level_abcd: int = 0
    per_sample_exact: list = field(default_factory=list)

    def update(self, pred: Optional[Tuple], gold_set: List[Tuple]):
        self.total += 1
        if pred is None or not gold_set:
            self.per_sample_exact.append(0.0)
            return
        self.valid_format += 1

        best_depth = max(hierarchical_match_depth(pred, g) for g in gold_set)
        is_exact = best_depth == 4
        self.per_sample_exact.append(float(is_exact))
        if is_exact:
            self.exact += 1
        if best_depth >= 1: self.level_a += 1
        if best_depth >= 2: self.level_ab += 1
        if best_depth >= 3: self.level_abc += 1
        if best_depth >= 4: self.level_abcd += 1

    def to_dict(self) -> dict:
        n = max(self.total, 1)
        exact_mean, exact_lo, exact_hi = bootstrap_ci(self.per_sample_exact)
        return {
            "n": self.total,
            "valid_format": round(self.valid_format / n, 4),
            "level_A": round(self.level_a / n, 4),
            "level_AB": round(self.level_ab / n, 4),
            "level_ABC": round(self.level_abc / n, 4),
            "level_ABCD": round(self.level_abcd / n, 4),
            "exact_match": round(self.exact / n, 4),
            "exact_match_ci95": [round(exact_lo, 4), round(exact_hi, 4)],
        }


@dataclass
class MultiSIDBeamMetrics:
    """Beam metrics for multi-SID gold: hit if ANY gold SID is in beam top-K."""
    k: int = 10
    total: int = 0
    per_sample_hit1: list = field(default_factory=list)
    per_sample_hit5: list = field(default_factory=list)
    per_sample_hit10: list = field(default_factory=list)
    per_sample_rr: list = field(default_factory=list)
    per_sample_coverage: list = field(default_factory=list)  # fraction of gold SIDs found

    def update(self, beam_preds: List[Optional[Tuple]], gold_set: List[Tuple]):
        self.total += 1
        if not gold_set:
            for lst in [self.per_sample_hit1, self.per_sample_hit5,
                        self.per_sample_hit10, self.per_sample_rr, self.per_sample_coverage]:
                lst.append(0.0)
            return

        gold_set_frozen = set(gold_set)

        # Hit@K: any gold SID in top-K predictions
        hit1 = float(len(beam_preds) > 0 and beam_preds[0] in gold_set_frozen)
        hit5 = float(any(p in gold_set_frozen for p in beam_preds[:5]))
        hit10 = float(any(p in gold_set_frozen for p in beam_preds[:10]))
        self.per_sample_hit1.append(hit1)
        self.per_sample_hit5.append(hit5)
        self.per_sample_hit10.append(hit10)

        # MRR: rank of first gold SID hit
        rr = 0.0
        for rank, pred in enumerate(beam_preds[:self.k], 1):
            if pred in gold_set_frozen:
                rr = 1.0 / rank
                break
        self.per_sample_rr.append(rr)

        # Coverage@K: fraction of gold SIDs found in beam
        beam_set = set(p for p in beam_preds[:self.k] if p is not None)
        found = len(gold_set_frozen & beam_set)
        self.per_sample_coverage.append(found / len(gold_set_frozen))

    def to_dict(self) -> dict:
        def _mean_ci(vals):
            m, lo, hi = bootstrap_ci(vals)
            return round(m, 4)

        h1_m, h1_lo, h1_hi = bootstrap_ci(self.per_sample_hit1)
        h5_m, h5_lo, h5_hi = bootstrap_ci(self.per_sample_hit5)
        h10_m, h10_lo, h10_hi = bootstrap_ci(self.per_sample_hit10)
        mrr_m, mrr_lo, mrr_hi = bootstrap_ci(self.per_sample_rr)
        cov_m, cov_lo, cov_hi = bootstrap_ci(self.per_sample_coverage)

        return {
            "n": self.total,
            "hit@1": round(h1_m, 4),
            "hit@5": round(h5_m, 4),
            "hit@10": round(h10_m, 4),
            "mrr@10": round(mrr_m, 4),
            "coverage@10": round(cov_m, 4),
            "coverage@10_ci95": [round(cov_lo, 4), round(cov_hi, 4)],
        }


# ============================================================================
# Text Generation Metrics
# ============================================================================

@dataclass
class TextGenMetrics:
    rouge_l_scores: list = field(default_factory=list)
    token_f1_scores: list = field(default_factory=list)
    token_jaccard_scores: list = field(default_factory=list)
    char_f1_scores: list = field(default_factory=list)
    cosine_sim_scores: list = field(default_factory=list)
    total: int = 0
    # Category-specific
    cat_scored: int = 0
    cat_exact_ref: int = 0
    cat_prefix_depth: list = field(default_factory=list)
    cat_level_match: dict = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0})

    def update(self, prediction: str, reference: str, task: str = ""):
        self.total += 1
        self.rouge_l_scores.append(rouge_l_f1(prediction, reference))
        self.token_f1_scores.append(token_f1(prediction, reference))
        self.token_jaccard_scores.append(token_jaccard(prediction, reference))
        self.char_f1_scores.append(char_f1(prediction, reference))

        if task == "sid_to_category" and reference:
            self.cat_scored += 1
            depth = category_prefix_depth(reference, prediction)
            self.cat_prefix_depth.append(depth)
            if reference.lower().strip() == prediction.lower().strip() and reference.strip():
                self.cat_exact_ref += 1
            for L in (1, 2, 3, 4):
                if category_level_match(reference, prediction, L):
                    self.cat_level_match[L] += 1

    def add_cosine_scores(self, scores: List[float]):
        self.cosine_sim_scores.extend(scores)

    def to_dict(self) -> dict:
        def _mean_ci(vals):
            m, lo, hi = bootstrap_ci(vals)
            return {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}

        result = {
            "n": self.total,
            "rouge_l": _mean_ci(self.rouge_l_scores),
            "token_f1": _mean_ci(self.token_f1_scores),
            "token_jaccard": _mean_ci(self.token_jaccard_scores),
            "char_f1": _mean_ci(self.char_f1_scores),
        }
        if self.cosine_sim_scores:
            result["cosine_sim"] = _mean_ci(self.cosine_sim_scores)

        if self.cat_scored > 0:
            cs = self.cat_scored
            result["category"] = {
                "n_scored": cs,
                "avg_prefix_depth": round(float(np.mean(self.cat_prefix_depth)), 3),
                "exact_ref_%": round(self.cat_exact_ref / cs * 100, 2),
                "level_match@1_%": round(self.cat_level_match[1] / cs * 100, 2),
                "level_match@2_%": round(self.cat_level_match[2] / cs * 100, 2),
                "level_match@3_%": round(self.cat_level_match[3] / cs * 100, 2),
                "level_match@4_%": round(self.cat_level_match[4] / cs * 100, 2),
            }

        return result


# ============================================================================
# Hallucination Tracker
# ============================================================================

class HallucinationTracker:
    def __init__(self):
        self.valid_sids: Set[Tuple] = set()
        self.total_generated = 0
        self.hallucinated = 0
        self.invalid_format = 0

    def load_corpus(self, val_df: pl.DataFrame):
        log.info("Building SID corpus for hallucination detection...")
        for conv in val_df["conversations"].to_list():
            for msg in conv:
                for m in SID_PATTERN.finditer(msg["content"]):
                    self.valid_sids.add(tuple(m.groups()))
        log.info(f"SID corpus: {len(self.valid_sids):,} unique valid SIDs")

    def check(self, pred_sid: Optional[Tuple]):
        self.total_generated += 1
        if pred_sid is None:
            self.invalid_format += 1
            return
        if pred_sid not in self.valid_sids:
            self.hallucinated += 1

    def to_dict(self) -> dict:
        n = max(self.total_generated, 1)
        valid_gen = self.total_generated - self.invalid_format
        return {
            "total_generated": self.total_generated,
            "invalid_format": self.invalid_format,
            "invalid_format_pct": round(self.invalid_format / n, 4),
            "hallucinated": self.hallucinated,
            "hallucination_rate": round(self.hallucinated / max(valid_gen, 1), 4),
            "corpus_size": len(self.valid_sids),
        }


# ============================================================================
# Error Analyzer
# ============================================================================

class ErrorAnalyzer:
    def __init__(self):
        self.level_a_confusion: Counter = Counter()
        self.first_wrong_level: Counter = Counter()
        self.gold_distribution: Counter = Counter()
        self.pred_distribution: Counter = Counter()
        self.total = 0
        self.invalid_format_count = 0

    def update(self, pred: Optional[Tuple], gold: Optional[Tuple]):
        self.total += 1
        if gold:
            self.gold_distribution[gold[0]] += 1
        if pred is None:
            self.invalid_format_count += 1
            self.first_wrong_level["invalid_format"] += 1
            return
        if gold is None:
            return
        self.pred_distribution[pred[0]] += 1

        if pred[0] != gold[0]:
            self.level_a_confusion[(gold[0], pred[0])] += 1
            self.first_wrong_level["A"] += 1
        elif pred[1] != gold[1]:
            self.first_wrong_level["B"] += 1
        elif pred[2] != gold[2]:
            self.first_wrong_level["C"] += 1
        elif pred[3] != gold[3]:
            self.first_wrong_level["D"] += 1
        else:
            self.first_wrong_level["correct"] += 1

    def to_dict(self) -> dict:
        n = max(self.total, 1)
        return {
            "total": self.total,
            "invalid_format_pct": round(self.invalid_format_count / n, 4),
            "first_wrong_level": {
                k: {"count": v, "pct": round(v / n, 4)}
                for k, v in self.first_wrong_level.most_common()
            },
            "top_level_A_confusions": [
                {"gold_A": g, "pred_A": p, "count": c}
                for (g, p), c in self.level_a_confusion.most_common(20)
            ],
            "gold_A_distribution_top10": [
                {"A": a, "count": c}
                for a, c in self.gold_distribution.most_common(10)
            ],
            "pred_A_distribution_top10": [
                {"A": a, "count": c}
                for a, c in self.pred_distribution.most_common(10)
            ],
        }


# ============================================================================
# Diversity Tracker
# ============================================================================

class DiversityTracker:
    def __init__(self, catalog_size: int = 0):
        self.generated_sids: List[str] = []
        self.catalog_size = catalog_size

    def add(self, sid_token: Optional[str]):
        if sid_token:
            self.generated_sids.append(sid_token)

    def to_dict(self) -> dict:
        total = len(self.generated_sids)
        if total == 0:
            return {"total_generated": 0}
        counts = Counter(self.generated_sids)
        unique = len(counts)
        top5_count = sum(c for _, c in counts.most_common(5))
        return {
            "total_generated": total,
            "unique_sids": unique,
            "unique_sid_rate_%": round(unique / total * 100, 2),
            "top_5_concentration_%": round(top5_count / total * 100, 2),
            "catalog_exploration_%": round(unique / max(self.catalog_size, 1) * 100, 3),
        }


# ============================================================================
# Qualitative Collector
# ============================================================================

class QualitativeCollector:
    def __init__(self, max_per_task: int = 10):
        self.max_per_task = max_per_task
        self.examples: Dict[str, list] = defaultdict(list)
        self._correct_count: Dict[str, int] = defaultdict(int)
        self._wrong_count: Dict[str, int] = defaultdict(int)

    def add_sid(self, task: str, input_text: str, gold_text: str,
                generated: str, gold_sid: Optional[Tuple], pred_sid: Optional[Tuple]):
        target = self.max_per_task // 2
        is_correct = pred_sid is not None and gold_sid is not None and pred_sid == gold_sid
        if is_correct and self._correct_count[task] >= target:
            return
        if not is_correct and self._wrong_count[task] >= target:
            return
        if is_correct:
            self._correct_count[task] += 1
        else:
            self._wrong_count[task] += 1

        match = "—"
        if pred_sid and gold_sid:
            depth = hierarchical_match_depth(pred_sid, gold_sid)
            match = "ABCD"[:depth] if depth > 0 else "none"

        self.examples[task].append({
            "input": input_text[:500],
            "gold": gold_text[:300],
            "generated": generated[:300],
            "gold_sid": list(gold_sid) if gold_sid else None,
            "pred_sid": list(pred_sid) if pred_sid else None,
            "correct": is_correct,
            "match_levels": match,
        })

    def add_text(self, task: str, input_text: str, gold: str,
                 generated: str, rouge_l: float):
        if len(self.examples[task]) >= self.max_per_task:
            return
        self.examples[task].append({
            "input": input_text[:500],
            "gold": gold[:500],
            "generated": generated[:500],
            "rouge_l": round(rouge_l, 4),
        })

    def to_dict(self) -> dict:
        return dict(self.examples)


# ============================================================================
# Data Loading
# ============================================================================

def resolve_data_dir(data_dir_arg: Optional[str]) -> Path:
    """Resolve data directory: CLI arg > env var > default (next to script)."""
    if data_dir_arg:
        p = Path(data_dir_arg).expanduser().resolve()
        if p.exists():
            return p

    env = os.environ.get("EVAL_DATA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    # Default: data/ next to this script
    script_dir = Path(__file__).parent
    default = script_dir / "data"
    if default.exists():
        return default

    if data_dir_arg:
        return Path(data_dir_arg).expanduser().resolve()
    raise FileNotFoundError("Data directory not found. Use --data-dir or set EVAL_DATA_DIR.")


def resolve_data_files(data_dir: Path) -> Tuple[Path, Path, Optional[Path]]:
    """Resolve paths: (val_file, sid_file, items_file).

    Supports three layouts:
    1. Canonical: data/semantic_llm_training/ + data/embeds/
    2. Flat with embeddings: all parquet files in one dir
    3. Flat minimal: val + sid only (no items_with_embeddings)
    """
    # Canonical layout
    canonical_val = data_dir / "semantic_llm_training" / "Pet_Supplies_conversations_val.parquet"
    canonical_sid = data_dir / "embeds" / "Pet_Supplies_items_with_semantic_ids.parquet"
    canonical_items = data_dir / "embeds" / "Pet_Supplies_items_with_embeddings_with_semantic_ids.parquet"
    if canonical_val.exists() and canonical_sid.exists():
        return (canonical_val, canonical_sid,
                canonical_items if canonical_items.exists() else None)

    # Flat layout
    flat_val = data_dir / "Pet_Supplies_conversations_val.parquet"
    flat_sid = data_dir / "Pet_Supplies_items_with_semantic_ids.parquet"
    flat_items = data_dir / "Pet_Supplies_items_with_embeddings_with_semantic_ids.parquet"
    if flat_val.exists() and flat_sid.exists():
        return (flat_val, flat_sid,
                flat_items if flat_items.exists() else None)

    raise FileNotFoundError(
        f"Required data files not found in {data_dir}. "
        "Need at least: Pet_Supplies_conversations_val.parquet + "
        "Pet_Supplies_items_with_semantic_ids.parquet"
    )


def build_catalog_mapping(
    sid_file: Path,
    items_file: Optional[Path],
) -> Tuple[Dict[Tuple, Dict[str, str]], Dict[int, str], Set[str]]:
    """Build:
    1. (A,B,C) -> {title, description_text, features_text, categories_text}
    2. A_level -> dominant category keyword
    3. Set of all valid SID token strings
    """
    sids_df = pl.read_parquet(str(sid_file))
    catalog_sids: Set[str] = set()
    a_level_category: Dict[int, str] = {}
    catalog_map: Dict[Tuple, Dict[str, str]] = {}

    for row in sids_df.iter_rows(named=True):
        sid = row.get("sid_tokens")
        if sid:
            catalog_sids.add(sid)

    if items_file and items_file.exists():
        items_df = pl.read_parquet(str(items_file))

        # Join sids with items for catalog mapping
        need_cols = {"semantic_ids", "title", "description_text", "features_text", "categories_text"}
        available = set(items_df.columns)

        if "semantic_ids" in available:
            # items_with_embeddings_with_semantic_ids has semantic_ids column
            for row in items_df.iter_rows(named=True):
                sid_list = row.get("semantic_ids")
                if not sid_list or len(sid_list) < 3:
                    continue
                key = (int(sid_list[0]), int(sid_list[1]), int(sid_list[2]))
                if key not in catalog_map:
                    catalog_map[key] = {
                        "title": row.get("title") or "",
                        "description_text": row.get("description_text") or "",
                        "features_text": row.get("features_text") or "",
                        "categories_text": row.get("categories_text") or "",
                        "item_context": row.get("item_context") or "",
                    }
        elif "parent_asin" in available and "parent_asin" in sids_df.columns:
            # Need to join sids with items
            joined = sids_df.join(items_df, on="parent_asin", how="left")
            a_level_cats = defaultdict(lambda: defaultdict(int))
            for row in joined.iter_rows(named=True):
                a = row.get("A")
                b = row.get("B")
                c = row.get("C")
                if a is None or b is None or c is None:
                    continue
                key = (int(a), int(b), int(c))
                if key not in catalog_map:
                    catalog_map[key] = {
                        "title": row.get("title") or "",
                        "description_text": row.get("description_text") or "",
                        "features_text": row.get("features_text") or "",
                        "categories_text": row.get("categories_text") or "",
                        "item_context": row.get("item_context") or "",
                    }
                title = (row.get("title") or "").lower()
                for kw, cat in CATEGORY_KEYWORDS.items():
                    if kw in title:
                        a_level_cats[int(a)][cat] += 1

            for a_level, cat_counts in a_level_cats.items():
                dominant, n = max(cat_counts.items(), key=lambda x: x[1])
                if n >= 3:
                    a_level_category[a_level] = dominant

    log.info(f"Catalog: {len(catalog_sids):,} SIDs, {len(catalog_map):,} mapped items, "
             f"{len(a_level_category)} A-levels with category")

    return catalog_map, a_level_category, catalog_sids


def load_eval_data(
    val_df: pl.DataFrame,
    samples_per_task: Optional[int],
    seed: int = 42,
) -> Dict[str, List[dict]]:
    """Sample and prepare evaluation data, grouped by task type.

    If samples_per_task is None, tier-specific defaults from TIER_SAMPLES
    are used (1000 for SID prediction, 500 for text generation).
    """
    rng = random.Random(seed)
    type_indices = defaultdict(list)
    for i, t in enumerate(val_df["type"].to_list()):
        type_indices[t].append(i)

    task_data: Dict[str, List[dict]] = {}
    for task_type, indices in sorted(type_indices.items()):
        if task_type not in ALL_KNOWN_TASKS:
            continue
        # Determine sample count: explicit override > tier default
        if samples_per_task is not None:
            n_samples = samples_per_task
        elif task_type in SID_PREDICTION_TASKS:
            n_samples = TIER_SAMPLES["sid_prediction"]
        elif task_type in TEXT_GENERATION_TASKS:
            n_samples = TIER_SAMPLES["text_generation"]
        else:
            n_samples = 200  # fallback
        chosen = rng.sample(indices, min(n_samples, len(indices)))
        items = []
        for idx in chosen:
            row = val_df.row(idx, named=True)
            conv = row["conversations"]
            gold_msg = conv[-1]
            if gold_msg["role"] != "assistant":
                continue
            prompt_msgs = [m for m in conv if m["role"] != "assistant"]
            items.append({
                "prompt_msgs": prompt_msgs,
                "gold": gold_msg["content"],
                "type": task_type,
                "user_content": next(
                    (m["content"] for m in conv if m["role"] == "user"), ""
                ),
                "system_content": next(
                    (m["content"] for m in conv if m["role"] == "system"), ""
                ),
            })
        if items:
            task_data[task_type] = items

    total = sum(len(v) for v in task_data.values())
    log.info(f"Loaded eval data: {total} samples across {len(task_data)} task types")
    for t, items in sorted(task_data.items()):
        log.info(f"  {t}: {len(items)} samples")
    return task_data


# ============================================================================
# Generation
# ============================================================================

def build_prompt(tokenizer, prompt_msgs: List[dict]) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        text = tokenizer.apply_chat_template(
            prompt_msgs, enable_thinking=False, **kwargs
        )
    except TypeError:
        text = tokenizer.apply_chat_template(prompt_msgs, **kwargs)
    return text.replace(_THINK_BLOCK, "")


def strip_generated(text: str) -> str:
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    text = text.replace(_THINK_BLOCK, "")
    return text.strip()


@torch.no_grad()
def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=False)


@torch.no_grad()
def generate_sampling(
    model, tokenizer, prompt: str,
    n_generations: int = 10,
    max_new_tokens: int = 48,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[str]:
    """Generate multiple candidates via sampling (like old eval_vast.py)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    results = []
    for _ in range(n_generations):
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated = out[0][inputs["input_ids"].shape[1]:]
        results.append(tokenizer.decode(generated, skip_special_tokens=False))
    return results


@torch.no_grad()
def generate_beam(
    model, tokenizer, prompt: str,
    num_beams: int = 10,
    max_new_tokens: int = 48,
    num_return_sequences: int = 10,
) -> List[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    try:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=min(num_return_sequences, num_beams),
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        log.warning(f"OOM in beam search (num_beams={num_beams}), falling back to greedy")
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return [tokenizer.decode(generated, skip_special_tokens=False)]

    input_len = inputs["input_ids"].shape[1]
    sequences = out.sequences
    seq_scores = getattr(out, "sequences_scores", None)

    order = list(range(sequences.shape[0]))
    if seq_scores is not None:
        order = sorted(order, key=lambda i: float(seq_scores[i]), reverse=True)

    return [
        tokenizer.decode(sequences[i][input_len:], skip_special_tokens=False)
        for i in order
    ]


# ============================================================================
# Perplexity (Catastrophic Forgetting)
# ============================================================================

@torch.no_grad()
def compute_perplexity(
    model, tokenizer,
    max_samples: int = 200,
    max_length: int = 512,
    stride: int = 256,
) -> dict:
    log.info("Computing perplexity on WikiText-2...")
    t0 = time.time()
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        log.warning(f"Failed to load WikiText-2: {e}")
        return {"error": str(e), "perplexity": None}

    texts = [t for t in ds["text"] if len(t.strip()) > 50][:max_samples]
    full_text = "\n\n".join(texts)
    encodings = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids = encodings["input_ids"].to(model.device)
    seq_len = input_ids.size(1)

    nlls = []
    n_tokens = 0
    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        target_begin = max(begin, begin if begin == 0 else begin + (max_length - stride))
        chunk_ids = input_ids[:, begin:end]
        target_ids = chunk_ids.clone()
        target_ids[:, :target_begin - begin] = -100
        outputs = model(input_ids=chunk_ids, labels=target_ids)
        nll = outputs.loss.float() * (target_ids != -100).sum().float()
        nlls.append(nll.item())
        n_tokens += (target_ids != -100).sum().item()
        if end >= seq_len:
            break

    ppl = math.exp(sum(nlls) / max(n_tokens, 1))
    elapsed = time.time() - t0
    log.info(f"WikiText-2 perplexity: {ppl:.2f} ({n_tokens:,} tokens, {elapsed:.0f}s)")
    return {"perplexity": round(ppl, 2), "n_tokens": n_tokens, "elapsed_s": round(elapsed, 1)}


# ============================================================================
# Performance Benchmarking
# ============================================================================

@torch.no_grad()
def benchmark_performance(
    model,
    tokenizer,
    prompts: List[str],
    beam_size: int = 10,
    max_new_tokens_sid: int = 48,
    max_new_tokens_text: int = 160,
    warmup_iters: int = 3,
    bench_iters: int = 20,
) -> dict:
    """Benchmark inference performance: TTFT, TPS, E2E latency, GPU memory, power.

    Args:
        prompts: list of formatted prompt strings (at least 1 needed)
        beam_size: beam width for beam search benchmark
        max_new_tokens_sid: max tokens for SID generation
        max_new_tokens_text: max tokens for text generation
        warmup_iters: warmup iterations (not measured)
        bench_iters: measured iterations

    Returns:
        dict with performance metrics
    """
    import subprocess

    device = next(model.parameters()).device
    if device.type != "cuda":
        log.warning("Performance benchmark requires CUDA, skipping")
        return {"error": "no CUDA device"}

    log.info(f"\n{'='*60}")
    log.info("PERFORMANCE BENCHMARK")
    log.info(f"{'='*60}")

    prompt = prompts[0]  # Use first prompt as representative
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    log.info(f"Benchmark prompt length: {input_len} tokens")

    results = {}

    # ---- GPU Memory: weights only ----
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    mem_weights_bytes = torch.cuda.memory_allocated(device)
    results["gpu_memory_weights_gb"] = round(mem_weights_bytes / 1e9, 2)

    # ---- GPU Power (average during benchmark) ----
    def _read_gpu_power() -> Optional[float]:
        """Read current GPU power draw in watts via nvidia-smi."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                 f"--id={device.index or 0}"],
                capture_output=True, text=True, timeout=5,
            )
            return float(out.stdout.strip().split("\n")[0])
        except Exception:
            return None

    # ---- 1. TTFT (Time to First Token) ----
    log.info("Measuring TTFT (prefill latency)...")

    # Warmup
    for _ in range(warmup_iters):
        _ = model(**inputs)

    ttft_times = []
    for _ in range(bench_iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = model(**inputs)
        torch.cuda.synchronize(device)
        ttft_times.append((time.perf_counter() - t0) * 1000)  # ms

    results["ttft_ms"] = {
        "mean": round(np.mean(ttft_times), 2),
        "std": round(np.std(ttft_times), 2),
        "min": round(np.min(ttft_times), 2),
        "max": round(np.max(ttft_times), 2),
        "n_iters": bench_iters,
        "input_tokens": input_len,
    }
    log.info(f"  TTFT: {results['ttft_ms']['mean']:.1f} ± {results['ttft_ms']['std']:.1f} ms")

    # ---- 2. E2E Greedy SID generation ----
    log.info("Measuring E2E greedy SID generation...")
    for _ in range(warmup_iters):
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens_sid,
                           do_sample=False, pad_token_id=tokenizer.pad_token_id)

    e2e_greedy_sid = []
    output_tokens_sid = []
    power_readings = []
    for _ in range(bench_iters):
        torch.cuda.synchronize(device)
        pw = _read_gpu_power()
        if pw:
            power_readings.append(pw)
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=max_new_tokens_sid,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        e2e_greedy_sid.append(elapsed * 1000)  # ms
        output_tokens_sid.append(out.shape[1] - input_len)

    avg_out_tokens_sid = np.mean(output_tokens_sid)
    avg_e2e_sid = np.mean(e2e_greedy_sid)
    tps_sid = avg_out_tokens_sid / (avg_e2e_sid / 1000) if avg_e2e_sid > 0 else 0

    results["e2e_greedy_sid_ms"] = {
        "mean": round(avg_e2e_sid, 1),
        "std": round(np.std(e2e_greedy_sid), 1),
        "min": round(np.min(e2e_greedy_sid), 1),
        "max": round(np.max(e2e_greedy_sid), 1),
        "avg_output_tokens": round(avg_out_tokens_sid, 1),
        "n_iters": bench_iters,
    }
    results["tps_sid"] = round(tps_sid, 1)
    log.info(f"  E2E greedy SID: {avg_e2e_sid:.0f} ms, "
             f"TPS: {tps_sid:.1f} tok/s, "
             f"avg output: {avg_out_tokens_sid:.0f} tokens")

    # ---- 3. E2E Greedy text generation ----
    log.info("Measuring E2E greedy text generation...")
    # Use a SID→text style prompt if possible, else same prompt with more tokens
    text_prompt = prompts[1] if len(prompts) > 1 else prompt
    text_inputs = tokenizer(text_prompt, return_tensors="pt").to(device)

    for _ in range(warmup_iters):
        _ = model.generate(**text_inputs, max_new_tokens=max_new_tokens_text,
                           do_sample=False, pad_token_id=tokenizer.pad_token_id)

    e2e_greedy_text = []
    output_tokens_text = []
    for _ in range(bench_iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = model.generate(**text_inputs, max_new_tokens=max_new_tokens_text,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        e2e_greedy_text.append(elapsed * 1000)
        output_tokens_text.append(out.shape[1] - text_inputs["input_ids"].shape[1])

    avg_out_tokens_text = np.mean(output_tokens_text)
    avg_e2e_text = np.mean(e2e_greedy_text)
    tps_text = avg_out_tokens_text / (avg_e2e_text / 1000) if avg_e2e_text > 0 else 0

    results["e2e_greedy_text_ms"] = {
        "mean": round(avg_e2e_text, 1),
        "std": round(np.std(e2e_greedy_text), 1),
        "min": round(np.min(e2e_greedy_text), 1),
        "max": round(np.max(e2e_greedy_text), 1),
        "avg_output_tokens": round(avg_out_tokens_text, 1),
        "n_iters": bench_iters,
    }
    results["tps_text"] = round(tps_text, 1)
    log.info(f"  E2E greedy text: {avg_e2e_text:.0f} ms, "
             f"TPS: {tps_text:.1f} tok/s, "
             f"avg output: {avg_out_tokens_text:.0f} tokens")

    # ---- 4. E2E Beam search ----
    if beam_size > 1:
        log.info(f"Measuring E2E beam search (k={beam_size})...")
        beam_iters = max(bench_iters // 2, 5)  # fewer iters — beam is slow

        for _ in range(min(warmup_iters, 2)):
            try:
                _ = model.generate(
                    **inputs, max_new_tokens=max_new_tokens_sid,
                    num_beams=beam_size, num_return_sequences=beam_size,
                    do_sample=False, early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log.warning("OOM during beam warmup, skipping beam benchmark")
                results["e2e_beam_ms"] = {"error": "OOM"}
                beam_size = 0  # skip beam
                break

        if beam_size > 1:
            e2e_beam = []
            for _ in range(beam_iters):
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                try:
                    _ = model.generate(
                        **inputs, max_new_tokens=max_new_tokens_sid,
                        num_beams=beam_size, num_return_sequences=beam_size,
                        do_sample=False, early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    torch.cuda.synchronize(device)
                    e2e_beam.append((time.perf_counter() - t0) * 1000)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    log.warning("OOM during beam benchmark iteration")
                    break

            if e2e_beam:
                results["e2e_beam_ms"] = {
                    "mean": round(np.mean(e2e_beam), 1),
                    "std": round(np.std(e2e_beam), 1),
                    "min": round(np.min(e2e_beam), 1),
                    "max": round(np.max(e2e_beam), 1),
                    "beam_size": beam_size,
                    "n_iters": len(e2e_beam),
                }
                log.info(f"  E2E beam k={beam_size}: {np.mean(e2e_beam):.0f} ms")

    # ---- 5. GPU Memory: peak (after beam search) ----
    peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    results["gpu_memory_peak_gb"] = round(peak_mem_bytes / 1e9, 2)
    log.info(f"  GPU Memory: weights={results['gpu_memory_weights_gb']:.1f} GB, "
             f"peak={results['gpu_memory_peak_gb']:.1f} GB")

    # ---- 6. GPU Power ----
    if power_readings:
        results["gpu_power_w"] = {
            "mean": round(np.mean(power_readings), 1),
            "min": round(np.min(power_readings), 1),
            "max": round(np.max(power_readings), 1),
        }
        # Energy per SID request
        energy_j = (np.mean(power_readings) * avg_e2e_sid / 1000)
        results["energy_per_sid_request_j"] = round(energy_j, 1)
        log.info(f"  GPU Power: {results['gpu_power_w']['mean']:.0f} W, "
                 f"Energy/SID: {energy_j:.1f} J")
    else:
        results["gpu_power_w"] = {"error": "nvidia-smi not available"}

    # ---- 7. Model info ----
    n_params = sum(p.numel() for p in model.parameters())
    results["model_parameters"] = n_params
    results["model_parameters_b"] = round(n_params / 1e9, 2)

    log.info(f"  Model: {results['model_parameters_b']:.2f}B parameters")
    log.info(f"{'='*60}")

    return results


# ============================================================================
# Intent Detection
# ============================================================================

def detect_intent_from_text(text: str) -> Optional[str]:
    t = text.lower()
    for kw, cat in CATEGORY_KEYWORDS.items():
        if kw in t:
            return cat
    return None


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_model(
    model_path: str,
    data_dir: str,
    model_name: str = "",
    samples_per_task: Optional[int] = None,
    beam_size: int = 10,
    max_new_tokens_sid: int = 48,
    max_new_tokens_text: int = 160,
    seed: int = 42,
    output_file: Optional[str] = None,
    attn_impl: str = "sdpa",
    skip_perplexity: bool = False,
    skip_cosine_sim: bool = False,
    qualitative_per_task: int = 10,
    dry_run: bool = False,
    resume: bool = False,
    decoding: str = "greedy",        # "greedy" or "sampling"
    n_generations: int = 10,         # for sampling mode
    temperature: float = 0.7,        # for sampling mode
    skip_benchmark: bool = False,
    bench_iters: int = 20,
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dry_run:
        samples_per_task = 2  # override tier defaults
        beam_size = min(beam_size, 3)
        skip_perplexity = True
        skip_cosine_sim = True
        log.info("DRY RUN: 2 samples/task, beam=3, no perplexity/cosine")

    # Log tier sampling config
    if samples_per_task is not None:
        log.info(f"Uniform sampling: {samples_per_task} samples/task")
    else:
        log.info(f"Tier sampling: SID={TIER_SAMPLES['sid_prediction']}, "
                 f"Text={TIER_SAMPLES['text_generation']}")

    # ---- Determine output path early (for resume) ----
    if output_file:
        out_path = Path(output_file)
    else:
        out_path = Path(model_path) / "eval_unified_results.json"

    # ---- Resume: load existing partial results ----
    completed_tasks: Set[str] = set()
    resumed_results: Optional[dict] = None
    if resume and out_path.exists():
        try:
            with open(out_path) as f:
                resumed_results = json.load(f)
            completed_tasks = set(resumed_results.get("tasks", {}).keys())
            if completed_tasks:
                log.info(f"RESUME: found {len(completed_tasks)} completed tasks: {sorted(completed_tasks)}")
        except Exception as e:
            log.warning(f"Could not load resume file: {e}")

    # ---- Load model ----
    log.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    # ---- Resolve data ----
    data_path = resolve_data_dir(data_dir)
    val_file, sid_file, items_file = resolve_data_files(data_path)
    log.info(f"Val: {val_file}")
    log.info(f"SIDs: {sid_file}")
    log.info(f"Items: {items_file}")

    val_df = pl.read_parquet(str(val_file))
    catalog_map, a_level_category, catalog_sids = build_catalog_mapping(sid_file, items_file)

    # ---- Perplexity ----
    ppl_result = {}
    if not skip_perplexity:
        ppl_result = compute_perplexity(model, tokenizer)

    # ---- Load eval data ----
    task_data = load_eval_data(val_df, samples_per_task, seed)

    # ---- Performance Benchmark ----
    bench_result = {}
    if not skip_benchmark and not dry_run:
        # Build representative prompts: one SID task, one text task
        bench_prompts = []
        for task_type in ["title_to_sid", "sid_to_title"]:
            if task_type in task_data and task_data[task_type]:
                item = task_data[task_type][0]
                bench_prompts.append(build_prompt(tokenizer, item["prompt_msgs"]))
        if not bench_prompts:
            # Fallback: use first available task
            for items in task_data.values():
                if items:
                    bench_prompts.append(build_prompt(tokenizer, items[0]["prompt_msgs"]))
                    break
        if bench_prompts:
            bench_result = benchmark_performance(
                model, tokenizer, bench_prompts,
                beam_size=beam_size,
                max_new_tokens_sid=max_new_tokens_sid,
                max_new_tokens_text=max_new_tokens_text,
                bench_iters=bench_iters,
            )
    elif dry_run:
        log.info("DRY RUN: skipping performance benchmark")

    # ---- Init trackers ----
    hallucination = HallucinationTracker()
    hallucination.load_corpus(val_df)
    global_error = ErrorAnalyzer()
    diversity = DiversityTracker(catalog_size=len(catalog_sids))
    qualitative = QualitativeCollector(max_per_task=qualitative_per_task)

    sim_backend = None
    if not skip_cosine_sim:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sim_backend = SimilarityBackend(device=device)

    results = {
        "meta": {
            "model_path": model_path,
            "model_name": model_name,
            "samples_per_task": samples_per_task if samples_per_task is not None else TIER_SAMPLES,
            "beam_size": beam_size,
            "decoding": decoding,
            "n_generations": n_generations if decoding == "sampling" else None,
            "temperature": temperature if decoding == "sampling" else None,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "similarity_backend": sim_backend.name() if sim_backend else "none",
        },
        "perplexity_wikitext2": ppl_result,
        "performance": bench_result,
        "tasks": {},
    }

    total_time = 0.0
    total_samples = 0

    # ---- Helper: save intermediate results ----
    def _save_intermediate():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- Per-task evaluation ----
    for task_type, items in sorted(task_data.items()):
        is_sid = task_type in SID_PREDICTION_TASKS
        is_text = task_type in TEXT_GENERATION_TASKS
        if not (is_sid or is_text):
            continue

        # ---- Resume: skip completed tasks ----
        if task_type in completed_tasks:
            results["tasks"][task_type] = resumed_results["tasks"][task_type]
            log.info(f"SKIP (resumed): {task_type}")
            continue

        log.info(f"\n{'='*60}")
        log.info(f"Evaluating: {task_type} ({len(items)} samples)")
        log.info(f"{'='*60}")
        t0 = time.time()

        if is_sid:
            greedy_metrics = SIDGreedyMetrics()
            beam_metrics = SIDBeamMetrics(k=beam_size)
            task_error = ErrorAnalyzer()
            intent_match_count = 0
            intent_total = 0

            for i, item in enumerate(tqdm(items, desc=task_type)):
                prompt = build_prompt(tokenizer, item["prompt_msgs"])
                gold_sid = parse_sid(item["gold"])

                # Greedy
                greedy_out = generate_greedy(model, tokenizer, prompt, max_new_tokens=max_new_tokens_sid)
                pred_sid = parse_sid(greedy_out)
                greedy_metrics.update(pred_sid, gold_sid)
                task_error.update(pred_sid, gold_sid)
                global_error.update(pred_sid, gold_sid)
                hallucination.check(pred_sid)

                if pred_sid:
                    diversity.add(sid_tuple_to_token(pred_sid))

                # Intent match
                query_intent = detect_intent_from_text(item["user_content"])
                if query_intent and pred_sid:
                    intent_total += 1
                    pred_a = int(pred_sid[0]) if pred_sid[0].isdigit() else None
                    if pred_a is not None and a_level_category.get(pred_a) == query_intent:
                        intent_match_count += 1

                qualitative.add_sid(
                    task_type, item["user_content"][:500], item["gold"],
                    strip_generated(greedy_out), gold_sid, pred_sid,
                )

                # Multi-candidate: beam search or sampling
                if decoding == "sampling":
                    sampling_outputs = generate_sampling(
                        model, tokenizer, prompt,
                        n_generations=n_generations,
                        max_new_tokens=max_new_tokens_sid,
                        temperature=temperature,
                    )
                    cand_sids = [parse_sid(out) for out in sampling_outputs]
                    cand_sids = [s for s in cand_sids if s is not None]
                    beam_metrics.update(cand_sids, gold_sid)
                elif beam_size > 1:
                    beam_outputs = generate_beam(
                        model, tokenizer, prompt,
                        num_beams=beam_size,
                        max_new_tokens=max_new_tokens_sid,
                        num_return_sequences=beam_size,
                    )
                    beam_sids = [parse_sid(out) for out in beam_outputs]
                    beam_sids = [s for s in beam_sids if s is not None]
                    beam_metrics.update(beam_sids, gold_sid)

            elapsed = time.time() - t0
            total_time += elapsed
            total_samples += len(items)

            task_result = {
                "type": "sid_prediction",
                "n": len(items),
                "greedy": greedy_metrics.to_dict(),
                "error_analysis": task_error.to_dict(),
                "elapsed_s": round(elapsed, 1),
                "samples_per_sec": round(len(items) / max(elapsed, 0.1), 2),
            }
            if decoding == "sampling" or beam_size > 1:
                label = "sampling" if decoding == "sampling" else "beam"
                task_result[label] = beam_metrics.to_dict()
            if intent_total > 0:
                task_result["intent_match"] = {
                    "matched": intent_match_count,
                    "total": intent_total,
                    "rate_%": round(intent_match_count / intent_total * 100, 2),
                }
            results["tasks"][task_type] = task_result
            _save_intermediate()

            g = greedy_metrics.to_dict()
            log.info(
                f"  greedy: valid={g['valid_format']:.1%} | "
                f"A={g['level_A']:.1%} | AB={g['level_AB']:.1%} | "
                f"ABC={g['level_ABC']:.1%} | exact={g['exact_match']:.1%}"
            )
            if decoding == "sampling" or beam_size > 1:
                b = beam_metrics.to_dict()
                lbl = "sampling" if decoding == "sampling" else "beam"
                log.info(
                    f"  {lbl}: hit@1={b['hit@1']:.1%} | hit@5={b['hit@5']:.1%} | "
                    f"hit@10={b['hit@10']:.1%} | mrr@10={b['mrr@10']:.4f} | "
                    f"ndcg@10={b['ndcg@10']:.4f}"
                )

        elif is_text:
            text_metrics = TextGenMetrics()
            pending_pairs: List[Tuple[str, str]] = []  # (response, gold) for cosine

            for i, item in enumerate(tqdm(items, desc=task_type)):
                prompt = build_prompt(tokenizer, item["prompt_msgs"])
                out = generate_greedy(model, tokenizer, prompt, max_new_tokens=max_new_tokens_text)
                prediction = strip_generated(out)

                # Get gold from catalog if available
                gold_text = item["gold"]
                sid_in_user = parse_sid(item["user_content"])
                if sid_in_user and task_type in TASK_TO_CATALOG_FIELD:
                    key = (int(sid_in_user[0]), int(sid_in_user[1]), int(sid_in_user[2]))
                    meta = catalog_map.get(key)
                    if meta:
                        catalog_gold = meta.get(TASK_TO_CATALOG_FIELD[task_type], "")
                        if catalog_gold:
                            gold_text = catalog_gold

                rl = rouge_l_f1(prediction, gold_text)
                text_metrics.update(prediction, gold_text, task=task_type)
                pending_pairs.append((prediction, gold_text))

                qualitative.add_text(
                    task_type, item["user_content"][:500], gold_text, prediction, rl,
                )

            # Cosine similarity (batch)
            if sim_backend and sim_backend.available() and pending_pairs:
                resps, golds = zip(*pending_pairs)
                cosine_scores = sim_backend.cosine_pairs(list(resps), list(golds))
                text_metrics.add_cosine_scores(cosine_scores)

            elapsed = time.time() - t0
            total_time += elapsed
            total_samples += len(items)

            task_result = {
                "type": "text_generation",
                "n": len(items),
                "text_metrics": text_metrics.to_dict(),
                "elapsed_s": round(elapsed, 1),
                "samples_per_sec": round(len(items) / max(elapsed, 0.1), 2),
            }
            results["tasks"][task_type] = task_result
            _save_intermediate()

            tm = text_metrics.to_dict()
            log.info(
                f"  rouge_l={tm['rouge_l']['mean']:.3f} | "
                f"token_f1={tm['token_f1']['mean']:.3f} | "
                f"jaccard={tm['token_jaccard']['mean']:.3f} | "
                f"char_f1={tm['char_f1']['mean']:.3f}"
            )
            if "cosine_sim" in tm:
                log.info(f"  cosine_sim={tm['cosine_sim']['mean']:.3f}")

    # ---- Aggregation by group ----
    results["groups"] = {}
    for group_name, group_tasks in TASK_GROUPS.items():
        present = [t for t in group_tasks if t in results["tasks"]]
        if not present:
            continue

        task_results = [results["tasks"][t] for t in present]
        if all(r["type"] == "sid_prediction" for r in task_results):
            # Aggregate SID metrics
            agg_greedy = SIDGreedyMetrics()
            agg_beam = SIDBeamMetrics(k=beam_size) if beam_size > 1 else None
            for r in task_results:
                g = r["greedy"]
                n = g["n"]
                agg_greedy.total += n
                agg_greedy.valid_format += round(g["valid_format"] * n)
                agg_greedy.level_a += round(g["level_A"] * n)
                agg_greedy.level_ab += round(g["level_AB"] * n)
                agg_greedy.level_abc += round(g["level_ABC"] * n)
                agg_greedy.level_abcd += round(g["level_ABCD"] * n)
                agg_greedy.exact += round(g["exact_match"] * n)
                agg_greedy.per_sample_exact.extend([g["exact_match"]] * n)
                agg_greedy.per_sample_a.extend([g["level_A"]] * n)

            group_result = {"greedy": agg_greedy.to_dict()}
            if agg_beam and all("beam" in r for r in task_results):
                # Merge beam per-sample lists (approximate via means)
                for r in task_results:
                    b = r["beam"]
                    n = b["n"]
                    for metric in ["hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10"]:
                        getattr(agg_beam, f"per_sample_{metric.replace('@', '')}",
                                agg_beam.per_sample_hit1)  # fallback
                # Just aggregate means for groups
                beam_agg = {}
                total_n = sum(r["beam"]["n"] for r in task_results)
                for metric in ["hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10"]:
                    weighted = sum(
                        r["beam"][metric] * r["beam"]["n"]
                        for r in task_results
                    ) / max(total_n, 1)
                    beam_agg[metric] = round(weighted, 4)
                beam_agg["n"] = total_n
                group_result["beam"] = beam_agg

            results["groups"][group_name] = group_result

        elif all(r["type"] == "text_generation" for r in task_results):
            # Aggregate text metrics (weighted mean)
            total_n = sum(r["text_metrics"]["n"] for r in task_results)
            agg = {}
            for metric in ["rouge_l", "token_f1", "token_jaccard", "char_f1"]:
                weighted = sum(
                    r["text_metrics"][metric]["mean"] * r["text_metrics"]["n"]
                    for r in task_results
                ) / max(total_n, 1)
                agg[metric] = round(weighted, 4)
            if all("cosine_sim" in r["text_metrics"] for r in task_results):
                weighted = sum(
                    r["text_metrics"]["cosine_sim"]["mean"] * r["text_metrics"]["n"]
                    for r in task_results
                ) / max(total_n, 1)
                agg["cosine_sim"] = round(weighted, 4)
            agg["n"] = total_n
            results["groups"][group_name] = agg

    # ---- Global summaries ----
    results["global_error_analysis"] = global_error.to_dict()
    results["hallucination"] = hallucination.to_dict()
    results["diversity"] = diversity.to_dict()
    results["total_time_s"] = round(total_time, 1)
    results["total_samples"] = total_samples

    # ---- Summary log ----
    log.info(f"\n{'='*70}")
    log.info("EVALUATION SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"Model: {model_path} ({model_name})")
    log.info(f"Total: {total_samples} samples in {total_time:.0f}s")
    if ppl_result.get("perplexity"):
        log.info(f"Perplexity (WikiText-2): {ppl_result['perplexity']:.2f}")
    if bench_result and "error" not in bench_result:
        log.info(f"TTFT: {bench_result.get('ttft_ms', {}).get('mean', '?')} ms")
        log.info(f"TPS (SID): {bench_result.get('tps_sid', '?')} tok/s")
        log.info(f"TPS (text): {bench_result.get('tps_text', '?')} tok/s")
        log.info(f"E2E greedy SID: {bench_result.get('e2e_greedy_sid_ms', {}).get('mean', '?')} ms")
        log.info(f"GPU Memory: {bench_result.get('gpu_memory_weights_gb', '?')} / {bench_result.get('gpu_memory_peak_gb', '?')} GB")
    h = results["hallucination"]
    log.info(f"Hallucination: {h['hallucination_rate']:.1%} ({h['hallucinated']}/{h['total_generated'] - h['invalid_format']})")
    d = results["diversity"]
    if d.get("total_generated"):
        log.info(f"Diversity: {d['unique_sids']} unique / {d['total_generated']} total ({d['unique_sid_rate_%']:.1f}%)")

    for gname, gm in results["groups"].items():
        if "greedy" in gm:
            g = gm["greedy"]
            line = f"  {gname}: A={g['level_A']:.1%} AB={g['level_AB']:.1%} exact={g['exact_match']:.1%}"
            if "beam" in gm:
                b = gm["beam"]
                line += f" | beam: h@10={b['hit@10']:.1%} mrr={b['mrr@10']:.4f} ndcg={b['ndcg@10']:.4f}"
            log.info(line)
        else:
            metrics_str = " | ".join(f"{k}={v:.3f}" for k, v in gm.items() if isinstance(v, float))
            log.info(f"  {gname}: {metrics_str}")

    # ---- Save ----
    if output_file:
        out_path = Path(output_file)
    else:
        out_path = Path(model_path) / "eval_unified_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"\nResults saved to {out_path}")

    # Save qualitative examples
    examples_path = out_path.with_name(out_path.stem + "_examples.json")
    with open(examples_path, "w") as f:
        json.dump(qualitative.to_dict(), f, indent=2, ensure_ascii=False)
    log.info(f"Examples saved to {examples_path}")

    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Unified evaluation of Semantic ID models for thesis"
    )
    p.add_argument("--model-path", required=True, help="Path to model checkpoint")
    p.add_argument("--data-dir", required=True, help="Path to data directory")
    p.add_argument("--model-name", default="", help="Model name label (e.g., '1.8B', '8B')")
    p.add_argument("--samples-per-task", type=int, default=None,
                   help="Override samples per task (default: tier-aware 1000/500)")
    p.add_argument("--beam-size", type=int, default=10)
    p.add_argument("--max-new-tokens-sid", type=int, default=48)
    p.add_argument("--max-new-tokens-text", type=int, default=160)
    p.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--output", default=None, help="Output JSON path")
    p.add_argument("--skip-perplexity", action="store_true")
    p.add_argument("--skip-cosine-sim", action="store_true")
    p.add_argument("--qualitative-per-task", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Quick test: 2 samples/task, small beam, no perplexity/cosine")
    p.add_argument("--resume", action="store_true",
                   help="Resume from partial results (skip already completed tasks)")
    p.add_argument("--decoding", default="greedy", choices=["greedy", "sampling"],
                   help="Decoding strategy for SID candidates (default: greedy+beam)")
    p.add_argument("--n-generations", type=int, default=10,
                   help="Number of sampling generations per sample (only with --decoding sampling)")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (only with --decoding sampling)")
    p.add_argument("--skip-benchmark", action="store_true",
                   help="Skip performance benchmarking (TTFT, TPS, E2E, GPU memory)")
    p.add_argument("--bench-iters", type=int, default=20,
                   help="Number of iterations for performance benchmark (default: 20)")

    args = p.parse_args()

    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        model_name=args.model_name,
        samples_per_task=args.samples_per_task,
        beam_size=args.beam_size,
        max_new_tokens_sid=args.max_new_tokens_sid,
        max_new_tokens_text=args.max_new_tokens_text,
        seed=args.seed,
        attn_impl=args.attn_impl,
        output_file=args.output,
        skip_perplexity=args.skip_perplexity,
        skip_cosine_sim=args.skip_cosine_sim,
        qualitative_per_task=args.qualitative_per_task,
        dry_run=args.dry_run,
        resume=args.resume,
        decoding=args.decoding,
        n_generations=args.n_generations,
        temperature=args.temperature,
        skip_benchmark=args.skip_benchmark,
        bench_iters=args.bench_iters,
    )
