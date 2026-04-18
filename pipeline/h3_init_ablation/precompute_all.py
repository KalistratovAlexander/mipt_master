#!/usr/bin/env python3
"""One-shot pre-registration artifacts for H3 ablation.

Produces three files under artifacts/ that must exist before any run_h3.sh:
  1. h3_init_scales.json — target Frobenius norm for scale-matched init.
  2. codebook.pt         — (768 × 32) RQ-VAE codebook for arm D.
  3. title_token_ids_per_sid.json — per-SID title-BPE multiset for arm C.

Usage:
  python precompute_all.py                   # all three steps
  python precompute_all.py --steps scales    # subset
  python precompute_all.py --force           # overwrite committed target_frobenius

Re-running is a no-op for scales (refuses to overwrite without --force); the
other two always regenerate.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from init_strategies import (  # noqa: E402
    N_CONTROL_TOKENS,
    N_NEW_TOTAL,
    N_SID_TOKENS,
    SID_CODES_PER_LEVEL,
    SID_LEVELS,
    compute_target_frobenius,
)


ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
_LEVEL_COLS = ["A", "B", "C", "D"][:SID_LEVELS]
_VQ_EMB_KEY = re.compile(r"vq_layers\.(\d+)\.embedding\.weight$")


# ---------------------------------------------------------------------------
# 1. Target Frobenius norm
# ---------------------------------------------------------------------------

def _git_sha(repo_dir: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def step_scales(args: argparse.Namespace) -> None:
    """Write target_frobenius into artifacts/h3_init_scales.json."""
    artifacts_path = Path(args.artifacts_path)
    with open(artifacts_path) as f:
        data = json.load(f)

    if data.get("target_frobenius") is not None and not args.force:
        print(f"[scales] target_frobenius already set to {data['target_frobenius']:.6f} — skip "
              "(use --force to overwrite; protocol violation if runs have started).")
        return

    if data["model_name"] != args.model_name:
        print(f"[scales] WARNING: model mismatch (JSON={data['model_name']!r}, "
              f"CLI={args.model_name!r}) — updating JSON.")
        data["model_name"] = args.model_name

    print(f"[scales] Loading {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, trust_remote_code=True,
    )
    E = model.get_input_embeddings().weight.detach()
    print(f"[scales] Embeddings: {tuple(E.shape)} dtype={E.dtype}")

    assert E.shape[1] == data["hidden_dim"], \
        f"hidden_dim mismatch: model={E.shape[1]}, JSON={data['hidden_dim']}"

    target = compute_target_frobenius(E, N_NEW_TOTAL, n_reps=args.n_reps, seed=args.slice_seed)
    # Split target into control / SID blocks so magnitudes are arm-invariant.
    # target_ctrl² + target_sid² == target² (Frobenius mass preserved).
    target_ctrl = target * math.sqrt(N_CONTROL_TOKENS / N_NEW_TOTAL)
    target_sid = target * math.sqrt(N_SID_TOKENS / N_NEW_TOTAL)
    print(f"[scales] target_frobenius = {target:.6f}  (over {args.n_reps} reps)")
    print(f"[scales]   target_frobenius_ctrl = {target_ctrl:.6f}  "
          f"({N_CONTROL_TOKENS}/{N_NEW_TOTAL} rows)")
    print(f"[scales]   target_frobenius_sid  = {target_sid:.6f}  "
          f"({N_SID_TOKENS}/{N_NEW_TOTAL} rows)")

    sha = _git_sha(Path(__file__).resolve().parents[2])
    data["target_frobenius"] = target
    data["target_frobenius_ctrl"] = target_ctrl
    data["target_frobenius_sid"] = target_sid
    data["n_reps_for_target"] = args.n_reps
    data["slice_seed"] = args.slice_seed
    data["committed_before_runs"] = True
    data["committed_git_sha"] = sha
    data["_status"] = (
        f"target_frobenius={target:.6f} from {args.model_name} via "
        f"compute_target_frobenius(E, {N_NEW_TOTAL}, seed={args.slice_seed}). "
        f"Split: ctrl={target_ctrl:.6f}, sid={target_sid:.6f}. "
        "COMMITTED — editing after first run is a protocol violation."
    )
    with open(artifacts_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"[scales] Wrote {artifacts_path}  (git SHA: {sha or 'unknown'})")


# ---------------------------------------------------------------------------
# 2. RQ-VAE codebook extraction (arm D)
# ---------------------------------------------------------------------------

def step_codebook(args: argparse.Namespace) -> None:
    """Extract (768 × 32) codebook from RQ-VAE checkpoint."""
    print(f"[codebook] Loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)

    per_level: dict[int, torch.Tensor] = {}
    for k, v in sd.items():
        m = _VQ_EMB_KEY.search(k)
        if m:
            per_level[int(m.group(1))] = v

    if not per_level:
        raise SystemExit(f"[codebook] No vq_layers.*.embedding.weight keys in {args.checkpoint}")

    n_levels = max(per_level) + 1
    if set(per_level) != set(range(n_levels)):
        raise SystemExit(f"[codebook] Missing levels: found {sorted(per_level)}, expected 0..{n_levels - 1}")

    for i in range(n_levels):
        if per_level[i].shape[0] != SID_CODES_PER_LEVEL:
            raise SystemExit(
                f"[codebook] Level {i}: {per_level[i].shape[0]} codes, expected {SID_CODES_PER_LEVEL}"
            )

    stacked = torch.cat([per_level[i] for i in range(n_levels)], dim=0).float()
    out = Path(args.codebook_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stacked, out)
    print(f"[codebook] Saved {tuple(stacked.shape)} to {out}  "
          f"Frobenius={torch.linalg.matrix_norm(stacked, ord='fro').item():.4f}")

    if n_levels < SID_LEVELS:
        missing_tokens = N_SID_TOKENS - n_levels * SID_CODES_PER_LEVEL
        print(f"[codebook]   NOTE: {SID_LEVELS - n_levels} level(s) beyond RQ-VAE "
              f"({missing_tokens} tokens) fall back to arm_A inside arm_D.")


# ---------------------------------------------------------------------------
# 3. Title-token map (arm C)
# ---------------------------------------------------------------------------

def step_title_map(args: argparse.Namespace) -> None:
    """Build per-SID multiset of BPE token ids from item titles."""
    print(f"[title-map] Loading items  : {args.items}")
    items = pd.read_parquet(args.items)[["parent_asin", args.title_col]]
    print(f"[title-map] Loading SIDs   : {args.sids}")
    sids = pd.read_parquet(args.sids)[["parent_asin", *_LEVEL_COLS]]
    df = items.merge(sids, on="parent_asin", how="inner")
    print(f"[title-map] Merged rows: {len(df):,}  (items={len(items):,}, sids={len(sids):,})")

    missing = df[args.title_col].isna().sum()
    if missing:
        print(f"[title-map] WARNING: {missing} items have null title — dropping")
        df = df.dropna(subset=[args.title_col])

    print(f"[title-map] Loading tokenizer: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    enc = tok(df[args.title_col].astype(str).tolist(), add_special_tokens=False)["input_ids"]

    buckets: list[list[int]] = [[] for _ in range(N_SID_TOKENS)]
    level_idx_bases = [lvl * SID_CODES_PER_LEVEL for lvl in range(SID_LEVELS)]
    levels = df[_LEVEL_COLS].to_numpy()
    for token_ids, row in zip(enc, levels):
        if not token_ids:
            continue
        for lvl, code in enumerate(row):
            buckets[level_idx_bases[lvl] + int(code)].extend(token_ids)

    empty = sum(1 for b in buckets if not b)
    total = sum(len(b) for b in buckets)
    out = Path(args.title_map_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(buckets, f)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"[title-map] Saved {N_SID_TOKENS} buckets ({empty} empty, {total:,} token occurrences) "
          f"→ {out} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

STEPS = {"scales": step_scales, "codebook": step_codebook, "title-map": step_title_map}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", default="scales,codebook,title-map",
                   help="Comma-separated subset of: scales, codebook, title-map")
    p.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--artifacts-path", default=str(ARTIFACTS_DIR / "h3_init_scales.json"))
    p.add_argument("--force", action="store_true",
                   help="[scales] overwrite target_frobenius (protocol violation post-run).")
    p.add_argument("--n-reps", type=int, default=32)
    p.add_argument("--slice-seed", type=int, default=0)
    p.add_argument("--checkpoint", default="models/rqvae/best_model.pth",
                   help="[codebook] RQ-VAE checkpoint path.")
    p.add_argument("--codebook-out", default=str(ARTIFACTS_DIR / "codebook.pt"))
    p.add_argument("--items", default="data/prepared/Pet_Supplies_items_cleaned.parquet",
                   help="[title-map] items parquet.")
    p.add_argument("--sids", default="data/embeds/Pet_Supplies_items_with_semantic_ids.parquet",
                   help="[title-map] item→SID parquet.")
    p.add_argument("--title-col", default="clean_title")
    p.add_argument("--title-map-out", default=str(ARTIFACTS_DIR / "title_token_ids_per_sid.json"))
    args = p.parse_args()

    requested = [s.strip() for s in args.steps.split(",") if s.strip()]
    for step in requested:
        if step not in STEPS:
            raise SystemExit(f"Unknown step: {step!r}. Known: {', '.join(STEPS)}")

    for step in requested:
        STEPS[step](args)

    print("\n>>> Done. Verify:")
    for name, path in [
        ("scales", args.artifacts_path),
        ("codebook", args.codebook_out),
        ("title-map", args.title_map_out),
    ]:
        print(f"  {name:10s} {path}")


if __name__ == "__main__":
    main()
