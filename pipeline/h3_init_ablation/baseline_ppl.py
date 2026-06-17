#!/usr/bin/env python3
"""Measure WikiText-2 perplexity on the *base* Qwen3-0.6B (no SID tokens).

Uses the same chunking (max_length=512, stride=256, max_samples=200) as
evaluate_unified.compute_perplexity so arm PPLs are directly comparable.

Output: JSON with {perplexity, n_tokens, elapsed_s, model_name}.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_ppl(model, tokenizer, max_samples=200, max_length=512, stride=256):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:max_samples]
    full_text = "\n\n".join(texts)
    enc = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"].to(model.device)
    seq_len = input_ids.size(1)
    nlls = []
    n_tokens = 0
    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        target_begin = begin if begin == 0 else begin + (max_length - stride)
        chunk_ids = input_ids[:, begin:end]
        target_ids = chunk_ids.clone()
        target_ids[:, : target_begin - begin] = -100
        out = model(input_ids=chunk_ids, labels=target_ids)
        nll = out.loss.float() * (target_ids != -100).sum().float()
        nlls.append(nll.item())
        n_tokens += (target_ids != -100).sum().item()
        if end >= seq_len:
            break
    ppl = math.exp(sum(nlls) / max(n_tokens, 1))
    return ppl, n_tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=200)
    args = p.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    mdl.eval()
    ppl, n_tok = compute_ppl(mdl, tok, max_samples=args.max_samples)
    elapsed = time.time() - t0
    result = {
        "model_name": args.model,
        "perplexity": round(ppl, 2),
        "n_tokens": int(n_tok),
        "elapsed_s": round(elapsed, 1),
        "max_samples": args.max_samples,
        "protocol": "wikitext-2-raw-v1 test; max_length=512 stride=256",
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
