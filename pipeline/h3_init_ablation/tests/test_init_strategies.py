"""Unit tests for H3 init strategies.

Critical invariants under test:
  - Shape correctness per arm.
  - Scale normalization: control and SID blocks each hit their registered target,
    and the control block is bit-identical across arms (not a confound).
  - Determinism: same seed → identical output.
  - Arm B has zero semantic signal (mean ≈ 0, variance ≈ σ²_E per-dim average).
  - Arm D projection P is orthonormal (P @ P.T = I_{d_code}).
  - Arm D is an isometry: codebook pairwise distances preserved.
  - Arm D partial-codebook path: first n_rq rows are projected, remaining
    N_SID_TOKENS - n_rq rows come from arm_A fallback (D is a collision-index
    counter in this project, so only 3 × 256 = 768 rows have learned codes).
  - Arm C falls back to arm A when title list is empty.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from init_strategies import (  # noqa: E402
    N_CONTROL_TOKENS,
    N_NEW_TOTAL,
    N_SID_TOKENS,
    arm_A,
    arm_B,
    arm_C,
    arm_D,
    build_new_rows,
    compute_target_frobenius,
    scale_normalize,
)


# Small fixtures — fast CPU-only tests
HIDDEN_DIM = 1024   # Qwen3-0.6B
V_EXISTING = 151_936
D_CODE = 32
N_RQVAE_LEVELS = 3                        # A/B/C learned; D is a collision counter
N_RQ_ROWS = N_RQVAE_LEVELS * 256          # 768 — matches this project's checkpoint


@pytest.fixture
def E_existing():
    torch.manual_seed(0)
    # Realistic scale: Qwen embeddings have ~0.02 std
    return torch.randn(V_EXISTING, HIDDEN_DIM) * 0.02


@pytest.fixture
def codebook():
    """Hypothetical full-coverage codebook (1024 rows) — exercises non-fallback path."""
    torch.manual_seed(0)
    return torch.randn(N_SID_TOKENS, D_CODE) * 0.1


@pytest.fixture
def codebook_partial():
    """Realistic codebook for this project: 3 RQ-VAE levels × 256 codes = 768 rows."""
    torch.manual_seed(0)
    return torch.randn(N_RQ_ROWS, D_CODE) * 0.1


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def test_arm_A_shape(E_existing):
    out = arm_A(E_existing, N_SID_TOKENS, seed=42)
    assert out.shape == (N_SID_TOKENS, HIDDEN_DIM)
    assert out.dtype == torch.float32


def test_arm_B_shape(E_existing):
    out = arm_B(E_existing, N_SID_TOKENS, seed=42)
    assert out.shape == (N_SID_TOKENS, HIDDEN_DIM)


def test_arm_C_shape(E_existing):
    title_tokens = [[0, 1, 2]] * N_SID_TOKENS
    fallback = arm_A(E_existing, N_SID_TOKENS, seed=42)
    out = arm_C(E_existing, title_tokens, fallback)
    assert out.shape == (N_SID_TOKENS, HIDDEN_DIM)


def test_arm_D_shape(codebook):
    out = arm_D(codebook, HIDDEN_DIM, seed=42)
    assert out.shape == (N_SID_TOKENS, HIDDEN_DIM)


def test_build_new_rows_total_shape(E_existing, codebook):
    title_tokens = [[0, 1, 2]] * N_SID_TOKENS
    for arm in ["A", "B", "C", "D"]:
        out = build_new_rows(
            arm=arm,
            E_existing=E_existing,
            seed=42,
            target_frobenius_ctrl=0.1,
            target_frobenius_sid=1.0,
            hidden_dim=HIDDEN_DIM,
            rqvae_codebook=codebook,
            title_token_ids_per_sid=title_tokens,
        )
        assert out.shape == (N_NEW_TOTAL, HIDDEN_DIM), f"arm {arm}: {out.shape}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm_fn,args", [
    ("A", ("E", N_SID_TOKENS, 42)),
    ("B", ("E", N_SID_TOKENS, 42)),
])
def test_determinism(arm_fn, args, E_existing):
    fn = {"A": arm_A, "B": arm_B}[arm_fn]
    out1 = fn(E_existing, args[1], args[2])
    out2 = fn(E_existing, args[1], args[2])
    assert torch.allclose(out1, out2)


def test_arm_D_determinism(codebook):
    a = arm_D(codebook, HIDDEN_DIM, seed=42)
    b = arm_D(codebook, HIDDEN_DIM, seed=42)
    assert torch.allclose(a, b)


def test_different_seeds_differ(E_existing):
    a = arm_A(E_existing, N_SID_TOKENS, seed=42)
    b = arm_A(E_existing, N_SID_TOKENS, seed=43)
    assert not torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Scale normalization — critical: all arms must end with identical Frobenius
# ---------------------------------------------------------------------------

def test_scale_normalize_hits_target():
    x = torch.randn(100, 50) * 5.0
    target = 7.5
    out = scale_normalize(x, target)
    actual = torch.linalg.matrix_norm(out, ord="fro").item()
    assert abs(actual - target) < 1e-3, f"target={target}, got {actual}"


def test_all_arms_same_frobenius_after_build(E_existing, codebook):
    """Key invariant: scale is NOT a confound between arms.

    With block-separate normalization both the control block and the SID block
    must hit their respective pre-registered Frobenius targets for every arm,
    so the total norm is identical across arms and magnitudes cannot bleed
    between blocks (which would happen under single-scalar normalize).
    """
    title_tokens = [[0, 1, 2]] * N_SID_TOKENS
    target_ctrl, target_sid = 0.3, 3.14
    for arm in ["A", "B", "C", "D"]:
        out = build_new_rows(
            arm=arm,
            E_existing=E_existing,
            seed=42,
            target_frobenius_ctrl=target_ctrl,
            target_frobenius_sid=target_sid,
            hidden_dim=HIDDEN_DIM,
            rqvae_codebook=codebook,
            title_token_ids_per_sid=title_tokens,
        )
        n_ctrl = torch.linalg.matrix_norm(out[:N_CONTROL_TOKENS], ord="fro").item()
        n_sid = torch.linalg.matrix_norm(out[N_CONTROL_TOKENS:], ord="fro").item()
        assert abs(n_ctrl - target_ctrl) < 1e-3, f"arm {arm} ctrl: {n_ctrl} != {target_ctrl}"
        assert abs(n_sid - target_sid) < 1e-3, f"arm {arm} sid: {n_sid} != {target_sid}"


def test_control_block_identical_across_arms(E_existing, codebook):
    """Control rows must be bit-identical across arms post-normalization.

    They're sampled from arm_A with the same seed, so pre-normalize they're the
    same; after block-separate normalize they're still the same (scale factor
    is a function of control_rows and target_ctrl only, both arm-invariant).
    """
    title_tokens = [[0, 1, 2]] * N_SID_TOKENS
    ctrls = {}
    for arm in ["A", "B", "C", "D"]:
        out = build_new_rows(
            arm=arm,
            E_existing=E_existing,
            seed=42,
            target_frobenius_ctrl=0.5,
            target_frobenius_sid=2.0,
            hidden_dim=HIDDEN_DIM,
            rqvae_codebook=codebook,
            title_token_ids_per_sid=title_tokens,
        )
        ctrls[arm] = out[:N_CONTROL_TOKENS].clone()
    for arm in ["B", "C", "D"]:
        assert torch.equal(ctrls["A"], ctrls[arm]), \
            f"control rows differ between A and {arm}"


def test_compute_target_frobenius_deterministic(E_existing):
    t1 = compute_target_frobenius(E_existing, N_NEW_TOTAL, seed=0)
    t2 = compute_target_frobenius(E_existing, N_NEW_TOTAL, seed=0)
    assert abs(t1 - t2) < 1e-6


# ---------------------------------------------------------------------------
# Arm B: variance-matched, zero mean
# ---------------------------------------------------------------------------

def test_arm_B_zero_mean(E_existing):
    out = arm_B(E_existing, n=10_000, seed=42)  # large n for stable estimate
    assert out.mean().abs() < 0.005


def test_arm_B_variance_matches_diag_cov(E_existing):
    E32 = E_existing.float()
    mu = E32.mean(dim=0)
    expected_sigma_sq = ((E32 - mu) ** 2).sum(dim=0) / (E32.shape[0] - 1)
    expected_avg = expected_sigma_sq.mean().item()

    out = arm_B(E_existing, n=10_000, seed=42)
    observed = out.var(unbiased=True).item()
    rel_err = abs(observed - expected_avg) / expected_avg
    assert rel_err < 0.05, f"expected σ²≈{expected_avg:.2e}, got {observed:.2e}"


# ---------------------------------------------------------------------------
# Arm D: orthonormal P, isometry of codebook
# ---------------------------------------------------------------------------

def test_arm_D_P_orthonormal(codebook):
    """Recover P from two codebook points and verify P @ P.T = I_{d_code}.

    Since out = codebook @ P, we can recover P via least squares:
    P = pinv(codebook) @ out.
    """
    out = arm_D(codebook, HIDDEN_DIM, seed=42)
    P_recovered = torch.linalg.pinv(codebook.float()) @ out
    PPt = P_recovered @ P_recovered.T
    I = torch.eye(D_CODE)
    err = (PPt - I).abs().max().item()
    assert err < 1e-3, f"P not orthonormal: max|PPt - I| = {err}"


def test_arm_D_preserves_pairwise_distances(codebook):
    """Isometry: ||C_i @ P - C_j @ P|| should equal ||C_i - C_j||."""
    out = arm_D(codebook, HIDDEN_DIM, seed=42)
    idx_a, idx_b = 10, 500
    dist_original = torch.linalg.vector_norm(codebook[idx_a].float() - codebook[idx_b].float()).item()
    dist_projected = torch.linalg.vector_norm(out[idx_a] - out[idx_b]).item()
    rel_err = abs(dist_projected - dist_original) / dist_original
    assert rel_err < 1e-4, f"isometry violated: {dist_original} → {dist_projected}"


def test_arm_D_requires_sufficient_hidden_dim(codebook):
    """P with orthonormal rows requires hidden_dim >= d_code."""
    with pytest.raises(AssertionError):
        arm_D(codebook, hidden_dim=16, seed=42)  # 16 < d_code=32


# ---------------------------------------------------------------------------
# Arm D — partial codebook (this project's reality: 3 RQ-VAE levels + D-counter)
# ---------------------------------------------------------------------------

def test_arm_D_partial_codebook_shape(codebook_partial, E_existing):
    out = arm_D(codebook_partial, HIDDEN_DIM, seed=42, E_existing=E_existing)
    assert out.shape == (N_SID_TOKENS, HIDDEN_DIM)


def test_arm_D_partial_preserves_codebook_rows(codebook_partial, E_existing):
    """First n_rq rows must be the projected codebook (identical to full-codebook path up to n_rq)."""
    out = arm_D(codebook_partial, HIDDEN_DIM, seed=42, E_existing=E_existing)
    # Reconstruct projection using the same seed
    gen = torch.Generator().manual_seed(42)
    G = torch.randn(HIDDEN_DIM, D_CODE, generator=gen, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    P = Q.T
    expected = codebook_partial.float() @ P
    assert torch.allclose(out[:N_RQ_ROWS], expected, atol=1e-5)


def test_arm_D_partial_fallback_matches_arm_A(codebook_partial, E_existing):
    """Fallback rows must equal arm_A(E_existing, n_fallback, seed + 7919)."""
    out = arm_D(codebook_partial, HIDDEN_DIM, seed=42, E_existing=E_existing)
    n_fallback = N_SID_TOKENS - N_RQ_ROWS
    expected = arm_A(E_existing, n_fallback, seed=42 + 7919)
    assert torch.allclose(out[N_RQ_ROWS:], expected, atol=1e-5)


def test_arm_D_partial_missing_E_existing_raises(codebook_partial):
    with pytest.raises(ValueError, match="E_existing"):
        arm_D(codebook_partial, HIDDEN_DIM, seed=42)  # no fallback source


def test_arm_D_partial_determinism(codebook_partial, E_existing):
    a = arm_D(codebook_partial, HIDDEN_DIM, seed=42, E_existing=E_existing)
    b = arm_D(codebook_partial, HIDDEN_DIM, seed=42, E_existing=E_existing)
    assert torch.allclose(a, b)


def test_build_new_rows_arm_D_partial_codebook(codebook_partial, E_existing):
    """build_new_rows must thread E_existing into arm_D when codebook is partial."""
    out = build_new_rows(
        arm="D",
        E_existing=E_existing,
        seed=42,
        target_frobenius_ctrl=0.1,
        target_frobenius_sid=1.0,
        hidden_dim=HIDDEN_DIM,
        rqvae_codebook=codebook_partial,
    )
    assert out.shape == (N_NEW_TOTAL, HIDDEN_DIM)
    n_sid = torch.linalg.matrix_norm(out[N_CONTROL_TOKENS:], ord="fro").item()
    assert abs(n_sid - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# Arm C: text-derived fallback
# ---------------------------------------------------------------------------

def test_arm_C_empty_title_falls_back(E_existing):
    """SIDs with no product-title tokens must use fallback rows."""
    title_tokens = [[] for _ in range(N_SID_TOKENS)]
    fallback = arm_A(E_existing, N_SID_TOKENS, seed=42)
    out = arm_C(E_existing, title_tokens, fallback)
    assert torch.allclose(out, fallback)


def test_arm_C_uses_mean_of_tokens(E_existing):
    """Non-empty title → mean of E_existing at those indices."""
    title_tokens = [[0, 1, 2]] * N_SID_TOKENS
    fallback = torch.zeros(N_SID_TOKENS, HIDDEN_DIM)
    out = arm_C(E_existing, title_tokens, fallback)
    expected = E_existing[:3].float().mean(dim=0)
    assert torch.allclose(out[0], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Validation: arm-wrong inputs surface errors early (before 54h of compute)
# ---------------------------------------------------------------------------

def test_build_unknown_arm_raises(E_existing):
    with pytest.raises(ValueError, match="unknown arm"):
        build_new_rows(
            arm="Z",
            E_existing=E_existing,
            seed=42,
            target_frobenius_ctrl=0.1,
            target_frobenius_sid=1.0,
            hidden_dim=HIDDEN_DIM,
        )


def test_build_C_without_titles_raises(E_existing):
    with pytest.raises(ValueError, match="title"):
        build_new_rows(
            arm="C",
            E_existing=E_existing,
            seed=42,
            target_frobenius_ctrl=0.1,
            target_frobenius_sid=1.0,
            hidden_dim=HIDDEN_DIM,
        )


def test_build_D_without_codebook_raises(E_existing):
    with pytest.raises(ValueError, match="codebook"):
        build_new_rows(
            arm="D",
            E_existing=E_existing,
            seed=42,
            target_frobenius_ctrl=0.1,
            target_frobenius_sid=1.0,
            hidden_dim=HIDDEN_DIM,
        )
