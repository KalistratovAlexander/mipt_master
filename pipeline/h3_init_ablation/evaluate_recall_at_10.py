#!/usr/bin/env python3
"""H3 primary-metric evaluator: Recall@10 on text→SID (title_to_sid task).

Loads a stage-2 model checkpoint, runs beam-search (beam=10) on the
`title_to_sid` subset of the held-out validation conversations, and writes
per-sample hit indicators + aggregate recall@1/5/10 to JSON. The per-sample
array is consumed by aggregate_stats.py for paired bootstrap contrasts.

Usage:
    python evaluate_recall_at_10.py \\
        --model-path /workspace/stage2/output/final \\
        --val-file data/semantic_llm_training/Pet_Supplies_conversations_val.parquet \\
        --n-samples 1000 \\
        --output runs/arm_A_seed_42/results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("h3-eval")

_SID_RE = re.compile(
    r"<\|sid_start\|><\|A(\d+)\|><\|B(\d+)\|><\|C(\d+)\|><\|D(\d+)\|><\|sid_end\|>"
)


def extract_sid(text: str) -> tuple[int, int, int, int] | None:
    m = _SID_RE.search(text)
    return tuple(int(g) for g in m.groups()) if m else None


def build_prompt(tokenizer, conversation: list[dict]) -> str:
    """Rebuild prompt up to (but excluding) the final assistant turn."""
    prefix = [m for m in conversation if m["role"] != "assistant"]
    try:
        return tokenizer.apply_chat_template(
            prefix, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument(
        "--val-file",
        default="data/semantic_llm_training/Pet_Supplies_conversations_val.parquet",
    )
    p.add_argument("--task", default="title_to_sid")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--beam-size", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--attn-impl", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--output", required=True)
    args = p.parse_args()

    import pandas as pd
    log.info(f"Loading {args.val_file}")
    df = pd.read_parquet(args.val_file)
    df = df[df["type"] == args.task].reset_index(drop=True)
    log.info(f"{args.task}: {len(df):,} rows")

    rng = random.Random(args.seed)
    chosen_idx = sorted(rng.sample(range(len(df)), min(args.n_samples, len(df))))
    df = df.iloc[chosen_idx].reset_index(drop=True)

    samples = []
    for row in df.itertuples(index=False):
        conv = list(row.conversations)
        gold_msg = next((m for m in conv if m["role"] == "assistant"), None)
        gold = extract_sid(gold_msg["content"]) if gold_msg else None
        if gold is None:
            continue
        samples.append({"conv": conv, "gold": gold})
    log.info(f"Parsed {len(samples):,} samples with valid gold SID")

    log.info(f"Loading {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    ).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        num_beams=args.beam_size,
        num_return_sequences=args.beam_size,
        do_sample=False,
        early_stopping=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.convert_tokens_to_ids("<|sid_end|>") or tok.eos_token_id,
    )

    per_sample_hits = {"hit@1": [], "hit@5": [], "hit@10": []}
    t0 = time.time()

    with torch.inference_mode():
        for i in tqdm(range(0, len(samples), args.batch_size)):
            batch = samples[i : i + args.batch_size]
            prompts = [build_prompt(tok, s["conv"]) for s in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model.generate(**enc, **gen_kwargs)
            # out: (B * beam, seq_len)
            out = out.reshape(len(batch), args.beam_size, -1)
            input_len = enc["input_ids"].shape[1]

            for s, beams in zip(batch, out):
                preds = []
                for b in beams:
                    text = tok.decode(b[input_len:], skip_special_tokens=False)
                    sid = extract_sid(text)
                    preds.append(sid)
                gold = s["gold"]
                per_sample_hits["hit@1"].append(int(gold in preds[:1]))
                per_sample_hits["hit@5"].append(int(gold in preds[:5]))
                per_sample_hits["hit@10"].append(int(gold in preds[:10]))

    elapsed = time.time() - t0
    n = len(per_sample_hits["hit@10"])
    recall = {k: sum(v) / n for k, v in per_sample_hits.items()}
    log.info(
        f"n={n}  hit@1={recall['hit@1']:.4f}  hit@5={recall['hit@5']:.4f}  "
        f"hit@10={recall['hit@10']:.4f}  ({elapsed:.1f}s)"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model_path": args.model_path,
            "task": args.task,
            "n_samples": n,
            "beam_size": args.beam_size,
            "seed": args.seed,
            "elapsed_seconds": elapsed,
            "recall@1": recall["hit@1"],
            "recall@5": recall["hit@5"],
            "recall@10": recall["hit@10"],
            "per_sample_hit@10": per_sample_hits["hit@10"],
        }, f, indent=2)
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
