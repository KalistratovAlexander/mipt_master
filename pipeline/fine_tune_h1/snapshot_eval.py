#!/usr/bin/env python3
"""Quick SID accuracy eval on a model snapshot — same metric as SIDEvalCallback,
but on N samples (default 1000), with batched greedy generation."""
import argparse
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_THINK_BLOCK = "<think>\n\n</think>\n\n"
_SID_PATTERN = re.compile(
    r"<\|sid_start\|>"
    r"<\|(A\d+)\|><\|(B\d+)\|><\|(C\d+)\|><\|(D\d+)\|>"
    r"<\|sid_end\|>"
)


def parse_sid(text):
    m = _SID_PATTERN.search(text)
    return tuple(m.groups()) if m else None


def apply_template(tokenizer, conversations):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        text = tokenizer.apply_chat_template(conversations, enable_thinking=False, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(conversations, **kwargs)
    return text.replace(_THINK_BLOCK, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params, vocab={model.config.vocab_size}")

    ds = load_dataset("parquet", data_files=args.val_file, split="train")
    reco_types = {"copurchase_forward", "copurchase_backward",
                  "seq_last_2", "seq_last_3", "seq_last_5"}
    reco_idx = [i for i, t in enumerate(ds["type"]) if t in reco_types]
    print(f"Reco samples in val: {len(reco_idx)}")

    import random
    rng = random.Random(args.seed)
    chosen = rng.sample(reco_idx, min(args.n_samples, len(reco_idx)))
    chosen.sort()

    items = []
    for idx in chosen:
        conv = list(ds[idx]["conversations"])
        if conv[-1]["role"] != "assistant":
            continue
        expected = conv[-1]["content"]
        exp_parsed = parse_sid(expected)
        if not exp_parsed:
            continue
        prompt_conv = conv[:-1]
        items.append({
            "prompt": apply_template(tokenizer, prompt_conv),
            "expected": exp_parsed,
            "type": ds[idx]["type"],
        })

    print(f"Eval items: {len(items)}")

    counts = {"total": 0, "valid": 0, "A": 0, "AB": 0, "ABC": 0, "exact": 0}
    by_type = {}
    t0 = time.time()

    for bi in range(0, len(items), args.batch_size):
        batch = items[bi:bi + args.batch_size]
        prompts = [it["prompt"] for it in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for i, it in enumerate(batch):
            gen_ids = out[i, enc["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            pred = parse_sid(text)
            counts["total"] += 1
            t = it["type"]
            by_type.setdefault(t, {"total": 0, "valid": 0, "A": 0, "AB": 0, "ABC": 0, "exact": 0})
            by_type[t]["total"] += 1
            if not pred:
                continue
            counts["valid"] += 1
            by_type[t]["valid"] += 1
            exp = it["expected"]
            if pred[0] == exp[0]:
                counts["A"] += 1; by_type[t]["A"] += 1
            if pred[:2] == exp[:2]:
                counts["AB"] += 1; by_type[t]["AB"] += 1
            if pred[:3] == exp[:3]:
                counts["ABC"] += 1; by_type[t]["ABC"] += 1
            if pred == exp:
                counts["exact"] += 1; by_type[t]["exact"] += 1

        elapsed = time.time() - t0
        done = counts["total"]
        eta = elapsed / done * (len(items) - done) if done else 0
        print(f"  [{done}/{len(items)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    n = counts["total"]
    print(f"\n=== Results (n={n}, time={time.time()-t0:.0f}s) ===")
    print(f"  valid_format: {100*counts['valid']/n:.1f}%")
    print(f"  level_A:      {100*counts['A']/n:.1f}%")
    print(f"  level_AB:     {100*counts['AB']/n:.1f}%")
    print(f"  level_ABC:    {100*counts['ABC']/n:.1f}%")
    print(f"  exact (Hit@1):{100*counts['exact']/n:.2f}%")

    print("\n=== By type ===")
    for t, c in sorted(by_type.items()):
        if c["total"]:
            print(f"  {t:30s} n={c['total']:4d}  valid={100*c['valid']/c['total']:5.1f}%  "
                  f"A={100*c['A']/c['total']:5.1f}%  exact={100*c['exact']/c['total']:5.2f}%")


if __name__ == "__main__":
    main()
