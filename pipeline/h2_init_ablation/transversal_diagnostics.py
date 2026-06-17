#!/usr/bin/env python3
"""H2 transversal diagnostics — geometry of SID-block embeddings after Stage 2.

Computes 4 diagnostic metrics on final-checkpoint weights (§3.1 of thesis/h2_metrics_plan.md):
  1. Pairwise cosine between arms (same SID, different arms, paired by seed).
  2. Linear CKA between arms (rotation-invariant sibling of #1).
  3. Effective rank of the SID block: exp(H(σ²_norm)) via SVD.
  4. RSA with RQ-VAE codebook: Pearson + Spearman between upper-triangular RDMs
     (SID block restricted to levels A/B/C = first 768 rows, matched against
     codebook of 768 codes × 32 dims).

Assumes:
  - Embedding matrix has N_NEW = 1027 new rows appended (3 control + 1024 SID).
    First 3 new rows are control tokens (<|rec|>, <|sid_start|>, <|sid_end|>);
    next 1024 are SID tokens ordered level-major: A_0..A_255, B_0..B_255,
    C_0..C_255, D_0..D_255 — see init_strategies.py header.
  - Codebook.pt is a (768, 32) tensor, same level-major order.

Usage:
  python transversal_diagnostics.py \\
      --runs-dir runs \\
      --codebook-path artifacts/codebook.pt \\
      --output results/transversal.json
"""
from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("h2-transversal")


N_CONTROL_TOKENS = 3
N_SID_TOKENS = 1024
N_NEW_TOTAL = N_CONTROL_TOKENS + N_SID_TOKENS
N_ABC_TOKENS = 768  # levels A+B+C — the portion with a codebook prior


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def effective_rank(E: torch.Tensor) -> float:
    """exp(H(σ²_norm)) from singular values of E. Roy & Vetterli 2007."""
    sigma = torch.linalg.svdvals(E.float())
    p = sigma.pow(2)
    p = p / p.sum()
    p = p.clamp_min(1e-12)
    H = -(p * p.log()).sum().item()
    return float(np.exp(H))


def pairwise_cosine_same_token(X: torch.Tensor, Y: torch.Tensor) -> float:
    """mean_i cos(X[i], Y[i]). X, Y must have same (n, h)."""
    assert X.shape == Y.shape, f"{X.shape} vs {Y.shape}"
    x = torch.nn.functional.normalize(X.float(), dim=1)
    y = torch.nn.functional.normalize(Y.float(), dim=1)
    return float((x * y).sum(dim=1).mean().item())


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA — rotation-invariant similarity. Kornblith 2019."""
    X = X.float() - X.float().mean(dim=0, keepdim=True)
    Y = Y.float() - Y.float().mean(dim=0, keepdim=True)
    num = (Y.T @ X).pow(2).sum()
    den = torch.linalg.matrix_norm(X.T @ X, ord="fro") * \
          torch.linalg.matrix_norm(Y.T @ Y, ord="fro")
    return float((num / den.clamp_min(1e-12)).item())


def upper_tri_rdm(M: torch.Tensor) -> np.ndarray:
    """Flatten upper-triangular (excl. diagonal) of RDM = 1 - cos(pairwise)."""
    X = torch.nn.functional.normalize(M.float(), dim=1)
    C = X @ X.T
    rdm = 1.0 - C
    n = rdm.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return rdm[iu[0], iu[1]].cpu().numpy()


def rsa_with_codebook(E_abc: torch.Tensor, codebook: torch.Tensor) -> dict:
    """Pearson + Spearman between RDM(SID-ABC) and RDM(codebook)."""
    assert E_abc.shape[0] == codebook.shape[0] == N_ABC_TOKENS, \
        f"expected {N_ABC_TOKENS} rows; got {E_abc.shape[0]}, {codebook.shape[0]}"
    rdm_sid = upper_tri_rdm(E_abc)
    rdm_code = upper_tri_rdm(codebook)
    return {
        "pearson": float(pearsonr(rdm_sid, rdm_code).statistic),
        "spearman": float(spearmanr(rdm_sid, rdm_code).statistic),
    }


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def load_sid_block(model_path: Path) -> torch.Tensor:
    """Return (N_SID_TOKENS, h) = last 1024 rows of input_embeddings."""
    log.info(f"Loading {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.float32, trust_remote_code=True,
    )
    E_full = model.get_input_embeddings().weight.data.detach().cpu().clone()
    del model
    E_new = E_full[-N_NEW_TOTAL:]
    E_sid = E_new[N_CONTROL_TOKENS : N_CONTROL_TOKENS + N_SID_TOKENS]
    assert E_sid.shape[0] == N_SID_TOKENS, f"got {E_sid.shape}"
    return E_sid


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default=str(Path(__file__).parent / "runs"))
    p.add_argument("--codebook-path",
                   default=str(Path(__file__).parent / "artifacts" / "codebook.pt"))
    p.add_argument("--output",
                   default=str(Path(__file__).parent / "results" / "transversal.json"))
    p.add_argument("--arms", default="A,B,C,D")
    p.add_argument("--seeds", default="42,43,44")
    args = p.parse_args()

    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    runs_dir = Path(args.runs_dir)

    # ---- Load codebook (768, 32) ----
    codebook = torch.load(args.codebook_path, map_location="cpu")
    if not torch.is_tensor(codebook):
        codebook = torch.as_tensor(codebook)
    codebook = codebook.float()
    assert codebook.shape == (N_ABC_TOKENS, 32), f"codebook shape {codebook.shape}"
    log.info(f"Codebook loaded: {tuple(codebook.shape)}")

    # ---- Load SID blocks for all (arm, seed) — keep in memory for cross-arm ----
    sid_blocks: dict[tuple[str, int], torch.Tensor] = {}
    for arm in arms:
        for sd in seeds:
            model_path = runs_dir / f"arm_{arm}_seed_{sd}" / "stage2" / "final"
            if not model_path.exists():
                log.warning(f"Missing: {model_path} — skipping")
                continue
            sid_blocks[(arm, sd)] = load_sid_block(model_path)

    if not sid_blocks:
        raise SystemExit("No models found under runs-dir")

    # ---- Per-(arm, seed): effective rank + RSA ----
    per_run: dict = {}
    for (arm, sd), E_sid in sid_blocks.items():
        E_abc = E_sid[:N_ABC_TOKENS]
        per_run[f"arm_{arm}_seed_{sd}"] = {
            "effective_rank": effective_rank(E_sid),
            "rsa_with_codebook": rsa_with_codebook(E_abc, codebook),
        }
        log.info(
            f"arm={arm} seed={sd}  "
            f"eff_rank={per_run[f'arm_{arm}_seed_{sd}']['effective_rank']:.2f}  "
            f"rsa_pearson={per_run[f'arm_{arm}_seed_{sd}']['rsa_with_codebook']['pearson']:.3f}  "
            f"rsa_spearman={per_run[f'arm_{arm}_seed_{sd}']['rsa_with_codebook']['spearman']:.3f}"
        )

    # ---- Per-arm: aggregate across seeds ----
    per_arm: dict = {}
    for arm in arms:
        ranks = [per_run[f"arm_{arm}_seed_{sd}"]["effective_rank"]
                 for sd in seeds if f"arm_{arm}_seed_{sd}" in per_run]
        pearsons = [per_run[f"arm_{arm}_seed_{sd}"]["rsa_with_codebook"]["pearson"]
                    for sd in seeds if f"arm_{arm}_seed_{sd}" in per_run]
        spearmans = [per_run[f"arm_{arm}_seed_{sd}"]["rsa_with_codebook"]["spearman"]
                     for sd in seeds if f"arm_{arm}_seed_{sd}" in per_run]
        per_arm[arm] = {
            "effective_rank_mean": float(np.mean(ranks)) if ranks else None,
            "effective_rank_std": float(np.std(ranks, ddof=1)) if len(ranks) > 1 else None,
            "rsa_pearson_mean": float(np.mean(pearsons)) if pearsons else None,
            "rsa_pearson_std": float(np.std(pearsons, ddof=1)) if len(pearsons) > 1 else None,
            "rsa_spearman_mean": float(np.mean(spearmans)) if spearmans else None,
            "rsa_spearman_std": float(np.std(spearmans, ddof=1)) if len(spearmans) > 1 else None,
        }

    # ---- Cross-arm: pairwise cosine + linear CKA, paired by seed ----
    cross_arm_per_seed: dict = {}
    for sd in seeds:
        available = [arm for arm in arms if (arm, sd) in sid_blocks]
        pair_results: dict = {}
        for a, b in combinations(available, 2):
            X, Y = sid_blocks[(a, sd)], sid_blocks[(b, sd)]
            pair_results[f"{a}_vs_{b}"] = {
                "pairwise_cosine": pairwise_cosine_same_token(X, Y),
                "linear_cka": linear_cka(X, Y),
            }
        cross_arm_per_seed[str(sd)] = pair_results

    # Average across seeds per pair
    cross_arm: dict = {}
    pair_keys = set()
    for sd_results in cross_arm_per_seed.values():
        pair_keys.update(sd_results.keys())
    for pk in sorted(pair_keys):
        cos_vals = [cross_arm_per_seed[str(sd)][pk]["pairwise_cosine"]
                    for sd in seeds if pk in cross_arm_per_seed[str(sd)]]
        cka_vals = [cross_arm_per_seed[str(sd)][pk]["linear_cka"]
                    for sd in seeds if pk in cross_arm_per_seed[str(sd)]]
        cross_arm[pk] = {
            "pairwise_cosine_mean": float(np.mean(cos_vals)),
            "pairwise_cosine_std": float(np.std(cos_vals, ddof=1)) if len(cos_vals) > 1 else None,
            "linear_cka_mean": float(np.mean(cka_vals)),
            "linear_cka_std": float(np.std(cka_vals, ddof=1)) if len(cka_vals) > 1 else None,
        }
        log.info(
            f"{pk}:  cos={cross_arm[pk]['pairwise_cosine_mean']:.3f}  "
            f"CKA={cross_arm[pk]['linear_cka_mean']:.3f}"
        )

    # ---- Write ----
    out = {
        "arms": arms,
        "seeds": seeds,
        "per_run": per_run,
        "per_arm": per_arm,
        "cross_arm_per_seed": cross_arm_per_seed,
        "cross_arm_mean": cross_arm,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
