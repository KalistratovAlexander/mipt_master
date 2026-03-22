"""Post-training evaluation for RQ-VAE.

Reconstruction quality, codebook utilization, collisions,
neighbor preservation, and prefix coincidence.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from .model import RQVAE

logger = logging.getLogger("train-rqvae")


# ═══════════════════════════════════════════════════════════════════════════
# Report data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReconstructionMetrics:
    mse: float
    cosine_mean: float
    cosine_p10: float
    cosine_p50: float
    cosine_p90: float


@dataclass
class CodebookLevelStats:
    level: int
    used_codes: int
    dead_codes: int
    perplexity: float
    top1_share: float


@dataclass
class CollisionStats:
    n_items: int
    n_unique: int
    collision_rate: float
    max_bucket_size: int


@dataclass
class EvalReport:
    reconstruction: ReconstructionMetrics
    codebook: list[CodebookLevelStats]
    collisions: CollisionStats
    nn_overlap: float
    prefix_coincidence: dict[int, tuple[float, float]] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_dataset(
    model: RQVAE, embeddings: Tensor, batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode all embeddings → (semantic_ids [N,L], reconstructions [N,D])."""
    model.eval()
    device = next(model.parameters()).device
    recon_list, sid_list = [], []
    for s in range(0, len(embeddings), batch_size):
        x = embeddings[s : s + batch_size].to(device)
        out = model(x)
        recon_list.append(out.x_recon.cpu().float().numpy())
        sid_list.append(model.encode_to_semantic_ids(x).cpu().numpy())
    return np.vstack(sid_list).astype(np.int64), np.vstack(recon_list)


def reconstruction_quality(E: np.ndarray, R: np.ndarray) -> ReconstructionMetrics:
    mse = float(np.mean((E - R) ** 2))
    cos = np.sum(E * R, axis=1) / (np.linalg.norm(E, axis=1) * np.linalg.norm(R, axis=1) + 1e-12)
    return ReconstructionMetrics(
        mse=mse, cosine_mean=float(cos.mean()),
        cosine_p10=float(np.quantile(cos, 0.1)),
        cosine_p50=float(np.quantile(cos, 0.5)),
        cosine_p90=float(np.quantile(cos, 0.9)),
    )


def codebook_stats(sids: np.ndarray, codebook_size: int) -> list[CodebookLevelStats]:
    stats = []
    for lvl in range(sids.shape[1]):
        vals, cnt = np.unique(sids[:, lvl], return_counts=True)
        p = cnt / cnt.sum()
        stats.append(CodebookLevelStats(
            level=lvl, used_codes=len(vals), dead_codes=codebook_size - len(vals),
            perplexity=float(np.exp(-(p * np.log(p + 1e-12)).sum())),
            top1_share=float(p.max()),
        ))
    return stats


def collision_stats(sids: np.ndarray) -> CollisionStats:
    cnt = Counter(map(tuple, sids))
    n = len(sids)
    return CollisionStats(
        n_items=n, n_unique=len(cnt),
        collision_rate=sum(v for v in cnt.values() if v > 1) / n,
        max_bucket_size=max(cnt.values()),
    )


def _topk_neighbors(E: np.ndarray, q_idx: np.ndarray, k: int) -> np.ndarray:
    sims = E[q_idx] @ E.T
    for i, q in enumerate(q_idx):
        sims[i, q] = -np.inf
    return np.argpartition(-sims, kth=k, axis=1)[:, :k]


def nn_overlap_at_k(
    E: np.ndarray, R: np.ndarray, k: int = 5, n_queries: int = 2000, seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    q = rng.choice(len(E), min(n_queries, len(E)), replace=False)
    R_norm = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    nn_e, nn_r = _topk_neighbors(E, q, k), _topk_neighbors(R_norm, q, k)
    overlap = sum(len(set(nn_e[i].tolist()) & set(nn_r[i].tolist())) for i in range(len(q)))
    return overlap / (len(q) * k)


def prefix_coincidence(
    E: np.ndarray, sids: np.ndarray, k: int = 10, n_queries: int = 2000, seed: int = 0,
) -> dict[int, tuple[float, float]]:
    """Prefix match rate among NN vs random. Returns {prefix_len: (nn_rate, random_rate)}."""
    rng = np.random.default_rng(seed)
    q = rng.choice(len(E), min(n_queries, len(E)), replace=False)
    nn = _topk_neighbors(E, q, k)
    nn_pairs = np.column_stack([np.repeat(q, k), nn.ravel()])
    rnd_pairs = rng.integers(0, len(E), size=(len(nn_pairs), 2))
    result = {}
    for p in range(1, sids.shape[1] + 1):
        nn_r = float(np.mean(np.all(sids[nn_pairs[:, 0], :p] == sids[nn_pairs[:, 1], :p], axis=1)))
        rr = float(np.mean(np.all(sids[rnd_pairs[:, 0], :p] == sids[rnd_pairs[:, 1], :p], axis=1)))
        result[p] = (nn_r, rr)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Top-level API
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_rqvae(
    model: RQVAE, embeddings: Tensor, codebook_size: int,
    batch_size: int = 512, k: int = 5, n_queries: int = 2000, seed: int = 0,
) -> EvalReport:
    E = embeddings.cpu().numpy().astype(np.float32)
    logger.info("Encoding dataset...")
    sids, recon = encode_dataset(model, embeddings, batch_size)
    logger.info("Computing metrics...")
    return EvalReport(
        reconstruction=reconstruction_quality(E, recon),
        codebook=codebook_stats(sids, codebook_size),
        collisions=collision_stats(sids),
        nn_overlap=nn_overlap_at_k(E, recon, k, n_queries, seed),
        prefix_coincidence=prefix_coincidence(E, sids, k * 2, n_queries, seed),
    )


def print_report(r: EvalReport) -> None:
    q = r.reconstruction
    print(f"\n{'='*60}")
    print(f"RECONSTRUCTION  MSE={q.mse:.6f}  cos: mean={q.cosine_mean:.4f} p10={q.cosine_p10:.4f} p50={q.cosine_p50:.4f} p90={q.cosine_p90:.4f}")
    print(f"CODEBOOK")
    for s in r.codebook:
        print(f"  L{s.level}: used={s.used_codes} dead={s.dead_codes} perplexity={s.perplexity:.1f} top1={s.top1_share:.4f}")
    c = r.collisions
    print(f"COLLISIONS  unique={c.n_unique:,}/{c.n_items:,} ({1-c.collision_rate:.1%})  max_bucket={c.max_bucket_size}")
    print(f"NN OVERLAP@k  {r.nn_overlap:.4f}")
    print(f"PREFIX COINCIDENCE (NN / Random)")
    for p, (nn, rnd) in sorted(r.prefix_coincidence.items()):
        print(f"  prefix {p}: {nn:.4f} / {rnd:.4f}")
    print(f"{'='*60}\n")
