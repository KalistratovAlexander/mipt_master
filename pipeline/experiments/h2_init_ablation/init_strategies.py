"""H2 4-way ablation: init strategies for new SID-token embedding rows.

Produces new_rows ∈ R^{n_new × h} that get written to the last n_new positions
of the embedding matrix after resize_token_embeddings().

Scale normalization is pre-registered (artifacts/h2_init_scales.json) and applied
per-block (control vs SID) so that neither scale is a confound between arms.

Convention: first 3 new tokens are control (<|rec|>, <|sid_start|>, <|sid_end|>);
remaining 1024 are SID-level (4 levels × 256 codes). For arms C and D the control
tokens fall back to arm A init (uniform across arms). The control and SID blocks
are Frobenius-normalized SEPARATELY — a single-scalar rescale on the concat would
let an arm's SID-block magnitude leak into its control tokens (breaks A↔D contrast).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch


N_CONTROL_TOKENS = 3
SID_LEVELS = 4
SID_CODES_PER_LEVEL = 256
N_SID_TOKENS = SID_LEVELS * SID_CODES_PER_LEVEL  # 1024
N_NEW_TOTAL = N_CONTROL_TOKENS + N_SID_TOKENS    # 1027


# ---------------------------------------------------------------------------
# Sampling primitives (all deterministic via torch.Generator)
# ---------------------------------------------------------------------------

def _sample_mvn(
    mu: torch.Tensor,
    cov: torch.Tensor,
    n: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample n vectors from N(mu, cov) via Cholesky. cov gets small jitter for PSD."""
    h = mu.shape[0]
    jitter = 1e-5 * torch.eye(h, dtype=cov.dtype, device=cov.device)
    L = torch.linalg.cholesky(cov + jitter)
    z = torch.randn(n, h, generator=generator, dtype=mu.dtype, device=mu.device)
    return mu.unsqueeze(0) + z @ L.T


def _empirical_mean_cov(E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """μ (h,), Σ (h, h) of existing rows. Computes in fp32 for numerical stability."""
    E32 = E.float()
    mu = E32.mean(dim=0)
    centered = E32 - mu
    cov = centered.T @ centered / (E32.shape[0] - 1)
    return mu, cov


# ---------------------------------------------------------------------------
# Arm A: N(μ_E, Σ_E) — SOTA default (OpenOneRec §4.2)
# ---------------------------------------------------------------------------

def arm_A(E_existing: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Full empirical covariance. Returns (n, h) in fp32."""
    gen = torch.Generator(device=E_existing.device).manual_seed(seed)
    mu, cov = _empirical_mean_cov(E_existing)
    return _sample_mvn(mu, cov, n, gen)


# ---------------------------------------------------------------------------
# Arm B: N(0, σ²·I) — variance-matched random control
# ---------------------------------------------------------------------------

def arm_B(E_existing: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Isotropic Gaussian with same average diagonal variance as Σ_E.

    Has same Frobenius scale as arm A (after normalization) but zero semantic
    structure. Tests whether init effects reduce to scale matching.
    """
    gen = torch.Generator(device=E_existing.device).manual_seed(seed)
    _, cov = _empirical_mean_cov(E_existing)
    sigma = torch.sqrt(torch.diagonal(cov).mean())
    h = E_existing.shape[1]
    z = torch.randn(n, h, generator=gen, dtype=torch.float32, device=E_existing.device)
    return z * sigma


# ---------------------------------------------------------------------------
# Arm C: mean of title-token embeddings — text-derived
# ---------------------------------------------------------------------------

def arm_C(
    E_existing: torch.Tensor,
    title_token_ids_per_sid: list[list[int]],
    fallback_rows: torch.Tensor,
) -> torch.Tensor:
    """Average embedding of product-title BPE tokens mapped to each SID.

    title_token_ids_per_sid[i] = list of BPE token ids from titles of products
    whose SID contains the i-th SID-level token. Empty list → fallback_rows[i].

    Args:
        E_existing: (V_old, h) existing embedding rows.
        title_token_ids_per_sid: len = N_SID_TOKENS (1024).
        fallback_rows: (N_SID_TOKENS, h) — arm A samples as fallback for empty lists.

    Returns (N_SID_TOKENS, h) in fp32.
    """
    assert len(title_token_ids_per_sid) == N_SID_TOKENS, \
        f"expected {N_SID_TOKENS} SID mappings, got {len(title_token_ids_per_sid)}"
    h = E_existing.shape[1]
    out = torch.empty(N_SID_TOKENS, h, dtype=torch.float32, device=E_existing.device)
    E32 = E_existing.float()
    for i, tok_ids in enumerate(title_token_ids_per_sid):
        if len(tok_ids) == 0:
            out[i] = fallback_rows[i]
        else:
            idx = torch.as_tensor(tok_ids, dtype=torch.long, device=E_existing.device)
            out[i] = E32[idx].mean(dim=0)
    return out


# ---------------------------------------------------------------------------
# Arm D: codebook-projected — Johnson-Lindenstrauss
# ---------------------------------------------------------------------------

def arm_D(
    rqvae_codebook: torch.Tensor,
    hidden_dim: int,
    seed: int,
    E_existing: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """v_i = C_i @ P where P ∈ R^{d_code × h} has orthonormal rows.

    Orthonormal rows preserve pairwise distances from codebook space exactly:
    ||x @ P - y @ P||² = ||x - y||² (isometry into a d_code-dim subspace of R^h).

    This project's RQ-VAE codebook only covers the first N_RQVAE_LEVELS levels
    (A/B/C = 3 × 256 = 768 codes). The 4th SID level D is a collision-index
    counter (standard TIGER-style design) — it has no learned codebook and thus
    no geometry to preserve. Tokens beyond len(rqvae_codebook) receive an arm_A
    MVN fallback and share the same scale-normalization pass.

    Args:
        rqvae_codebook: (n_rq, d_code) with n_rq ≤ N_SID_TOKENS. Codewords
                        stacked level-major: [A_0..A_255, B_0..B_255, C_0..C_255, ...].
        hidden_dim: h (e.g., 1024 for Qwen3-0.6B).
        seed: controls random orthonormal P AND the arm_A fallback sampling.
        E_existing: required when n_rq < N_SID_TOKENS — used for fallback MVN.

    Returns (N_SID_TOKENS, h) in fp32.
    """
    n_rq = rqvae_codebook.shape[0]
    d_code = rqvae_codebook.shape[1]
    assert n_rq <= N_SID_TOKENS, f"codebook has {n_rq} rows > N_SID_TOKENS={N_SID_TOKENS}"
    assert hidden_dim >= d_code, \
        f"hidden_dim={hidden_dim} must be >= d_code={d_code} for orthonormal P"

    gen = torch.Generator().manual_seed(seed)
    G = torch.randn(hidden_dim, d_code, generator=gen, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)                       # (h, d_code), orthonormal cols
    P = Q.T                                         # (d_code, h), orthonormal rows
    projected = rqvae_codebook.float() @ P          # (n_rq, h)

    if n_rq == N_SID_TOKENS:
        return projected

    if E_existing is None:
        raise ValueError(
            f"arm_D: codebook provides {n_rq} rows, {N_SID_TOKENS - n_rq} still "
            "needed — pass E_existing for arm_A fallback."
        )
    n_fallback = N_SID_TOKENS - n_rq
    fallback = arm_A(E_existing, n_fallback, seed + 7919).to(projected.device)
    return torch.cat([projected, fallback], dim=0)


# ---------------------------------------------------------------------------
# Scale normalization (pre-registered)
# ---------------------------------------------------------------------------

def compute_target_frobenius(
    E_existing: torch.Tensor,
    n_new: int,
    n_reps: int = 32,
    seed: int = 0,
) -> float:
    """Expected Frobenius norm of a random n_new-row slice of E_existing.

    Deterministic given (E_existing, seed). Value is committed to
    artifacts/h2_init_scales.json before any training run.
    """
    gen = torch.Generator(device=E_existing.device).manual_seed(seed)
    V = E_existing.shape[0]
    E32 = E_existing.float()
    norms = []
    for _ in range(n_reps):
        idx = torch.randint(0, V, (n_new,), generator=gen, device=E_existing.device)
        norms.append(torch.linalg.matrix_norm(E32[idx], ord="fro").item())
    return float(sum(norms) / len(norms))


def scale_normalize(rows: torch.Tensor, target_frobenius: float) -> torch.Tensor:
    """Rescale rows so ||rows||_F == target_frobenius."""
    current = torch.linalg.matrix_norm(rows.float(), ord="fro").item()
    if current < 1e-8:
        raise ValueError(
            f"near-zero Frobenius norm ({current:.2e}); cannot rescale"
        )
    return rows * (target_frobenius / current)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_new_rows(
    arm: str,
    E_existing: torch.Tensor,
    seed: int,
    target_frobenius_ctrl: float,
    target_frobenius_sid: float,
    hidden_dim: int,
    rqvae_codebook: Optional[torch.Tensor] = None,
    title_token_ids_per_sid: Optional[list[list[int]]] = None,
) -> torch.Tensor:
    """Produce (N_NEW_TOTAL, h) initialization for all new tokens under chosen arm.

    Flow:
      1. Sample the 3 control tokens from arm A (identical across arms).
      2. Sample arm-specific rows for the 1024 SID-level tokens.
      3. Scale-normalize the two blocks SEPARATELY, then concatenate.

    Target split: target_ctrl² + target_sid² = target_total² (mass preserved).
    Under uniform-variance assumption target_ctrl/target_sid ≈ √(3/1027)/√(1024/1027).

    Returns (1027, h) in fp32 on the same device as E_existing.
    """
    # Control tokens — arm A for all (isolation from SID-init question)
    control_rows = arm_A(E_existing, N_CONTROL_TOKENS, seed)
    ctrl_scaled = scale_normalize(control_rows, target_frobenius_ctrl)

    # SID-level tokens
    if arm == "A":
        sid_rows = arm_A(E_existing, N_SID_TOKENS, seed + 1)
    elif arm == "B":
        sid_rows = arm_B(E_existing, N_SID_TOKENS, seed + 1)
    elif arm == "C":
        if title_token_ids_per_sid is None:
            raise ValueError("arm C requires title_token_ids_per_sid")
        fallback = arm_A(E_existing, N_SID_TOKENS, seed + 1)
        sid_rows = arm_C(E_existing, title_token_ids_per_sid, fallback)
    elif arm == "D":
        if rqvae_codebook is None:
            raise ValueError("arm D requires rqvae_codebook")
        sid_rows = arm_D(
            rqvae_codebook, hidden_dim, seed + 1, E_existing=E_existing,
        ).to(E_existing.device)
    else:
        raise ValueError(f"unknown arm: {arm!r}; expected one of A/B/C/D")

    sid_scaled = scale_normalize(sid_rows, target_frobenius_sid)
    new_rows = torch.cat([ctrl_scaled, sid_scaled], dim=0)
    assert new_rows.shape == (N_NEW_TOTAL, hidden_dim)
    return new_rows


def apply_init_to_model(
    model,
    arm: str,
    seed: int,
    target_frobenius_ctrl: float,
    target_frobenius_sid: float,
    rqvae_codebook: Optional[torch.Tensor] = None,
    title_token_ids_per_sid: Optional[list[list[int]]] = None,
) -> None:
    """Write H2 init into model's last N_NEW_TOTAL embedding rows in place.

    Pre-conditions:
      - model.resize_token_embeddings(V_old + N_NEW_TOTAL) already called.
      - Model is on CPU or single GPU (no sharding assumed).
    """
    in_emb = model.get_input_embeddings()
    W = in_emb.weight
    h = W.shape[1]
    E_existing = W[:-N_NEW_TOTAL].detach().clone()

    new_rows = build_new_rows(
        arm=arm,
        E_existing=E_existing,
        seed=seed,
        target_frobenius_ctrl=target_frobenius_ctrl,
        target_frobenius_sid=target_frobenius_sid,
        hidden_dim=h,
        rqvae_codebook=rqvae_codebook,
        title_token_ids_per_sid=title_token_ids_per_sid,
    )

    with torch.no_grad():
        W[-N_NEW_TOTAL:] = new_rows.to(W.dtype).to(W.device)
        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight is not W:
            # Untied — out of H2 scope, but keep safe default: mirror input.
            out_emb.weight[-N_NEW_TOTAL:] = new_rows.to(out_emb.weight.dtype).to(out_emb.weight.device)


# ---------------------------------------------------------------------------
# Pre-registration I/O
# ---------------------------------------------------------------------------

def load_pre_registered(path: Path | str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_pre_registered(path: Path | str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
