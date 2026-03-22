#!/usr/bin/env python3
"""Stage 1: Vocabulary expansion for Qwen3-1.8B with Semantic IDs.

Adds 1027 SID tokens to vocabulary and trains ONLY their embeddings.
All other params frozen. Qwen3-1.8B has tied embeddings (input == output).

1.8B-specific: batch=64, no gradient_checkpointing.

Based on OpenOneRec Stage 1 (arxiv:2512.24762).
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage1-1.8b")

_THINK_BLOCK = "<think>\n\n</think>\n\n"

SID_LEVELS = ["A", "B", "C", "D"]
SID_CODEBOOK_SIZE = 256


# ---------------------------------------------------------------------------
# Instruction masking
# ---------------------------------------------------------------------------

def _get_masking_ids(tokenizer) -> tuple[list[int], int]:
    template_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    return template_ids, im_end_id


def _apply_chat_mask(input_ids: list[int], template_ids: list[int], im_end_id: int) -> list[int]:
    """Labels: -100 for everything except assistant responses (multi-turn)."""
    r = len(template_ids)
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if i + r <= len(input_ids) and input_ids[i : i + r] == template_ids:
            j = i + r
            while j < len(input_ids):
                labels[j] = input_ids[j]
                if input_ids[j] == im_end_id:
                    j += 1
                    break
                j += 1
            i = j
        else:
            i += 1
    return labels


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class DataCollatorForCausalLM:
    """Pads batch preserving pre-computed labels with -100 masks."""

    def __init__(self, tokenizer, pad_to_multiple_of: int = 64):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict]) -> dict:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        padded_labels = [lb + [-100] * (max_len - len(lb)) for lb in labels]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


# ---------------------------------------------------------------------------
# Vocabulary extension
# ---------------------------------------------------------------------------

def make_sid_tokens() -> list[str]:
    tokens = ["<|rec|>", "<|sid_start|>", "<|sid_end|>"]
    for level in SID_LEVELS:
        tokens += [f"<|{level}{i}|>" for i in range(SID_CODEBOOK_SIZE)]
    return tokens  # 3 + 4*256 = 1027


def _init_new_rows(weight: torch.Tensor, n_new: int) -> None:
    """Initialize last n_new rows from N(mean, std) of existing embeddings."""
    existing = weight[:-n_new]
    mean = existing.mean(dim=0)
    std = existing.std(dim=0).clamp(min=1e-6)
    weight[-n_new:] = mean + torch.randn(
        n_new, weight.shape[1], dtype=weight.dtype, device=weight.device
    ) * std


def extend_vocabulary(model, tokenizer) -> int:
    tokens = make_sid_tokens()
    n_before = len(tokenizer)
    tokenizer.add_tokens(tokens, special_tokens=True)
    n_new = len(tokenizer) - n_before
    if n_new == 0:
        log.info("SID tokens already in vocab")
        return 0

    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        in_emb = model.get_input_embeddings()
        _init_new_rows(in_emb.weight, n_new)

        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight is not in_emb.weight:
            _init_new_rows(out_emb.weight, n_new)
            log.info(f"Untied: both matrices initialized ({n_new} tokens)")
        else:
            log.info(f"Tied: single matrix initialized ({n_new} tokens)")

    log.info(f"Vocab: {n_before:,} -> {len(tokenizer):,} (+{n_new} SID tokens)")
    return n_new


def freeze_except_embeddings(model) -> None:
    """Freeze all params except embedding matrices."""
    for p in model.parameters():
        p.requires_grad = False

    in_emb = model.get_input_embeddings()
    in_emb.weight.requires_grad = True

    out_emb = model.get_output_embeddings()
    untied = out_emb is not None and out_emb.weight is not in_emb.weight
    if untied:
        out_emb.weight.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        f"{'Untied' if untied else 'Tied'} embeddings | "
        f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _apply_template(tokenizer, conversations) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=False)
    try:
        text = tokenizer.apply_chat_template(conversations, enable_thinking=False, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(conversations, **kwargs)
    return text.replace(_THINK_BLOCK, "")


def load_and_tokenize(
    path: str, tokenizer, max_samples: int, max_length: int, seed: int,
    template_ids: list[int], im_end_id: int,
):
    ds = load_dataset("parquet", data_files=path, split="train")
    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))

    def process(batch):
        texts = [_apply_template(tokenizer, conv) for conv in batch["conversations"]]
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        enc["labels"] = [
            _apply_chat_mask(ids, template_ids, im_end_id)
            for ids in enc["input_ids"]
        ]
        return enc

    ds = ds.map(process, batched=True, remove_columns=ds.column_names, num_proc=4,
                desc=f"Tokenizing {Path(path).name}")
    log.info(f"Dataset: {len(ds):,} examples from {Path(path).name}")
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Stage 1: Vocab expansion (Qwen3-1.8B)")
    p.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--train-file", required=True)
    p.add_argument("--val-file", default=None)
    p.add_argument("--output-dir", default="output/stage1_1.8b")

    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--max-train-samples", type=int, default=64_000)
    p.add_argument("--max-val-samples", type=int, default=2_000)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--no-torch-compile", action="store_true")
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    log.info(f"Loading {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    log.info(f"Attention: {attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    n_new = extend_vocabulary(model, tokenizer)
    freeze_except_embeddings(model)

    template_ids, im_end_id = _get_masking_ids(tokenizer)

    train_ds = load_and_tokenize(
        args.train_file, tokenizer, args.max_train_samples,
        args.max_seq_length, args.seed, template_ids, im_end_id,
    )
    val_ds = (
        load_and_tokenize(
            args.val_file, tokenizer, args.max_val_samples,
            args.max_seq_length, args.seed, template_ids, im_end_id,
        )
        if args.val_file and Path(args.val_file).exists()
        else None
    )

    use_compile = not args.no_torch_compile
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForCausalLM(tokenizer),
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            # weight_decay=0 for embeddings: L2 pulls ALL rows toward zero each step,
            # including tokens absent from batch — kills new SID embeddings before they learn
            weight_decay=0.0,
            bf16=True,
            gradient_checkpointing=False,
            optim="adamw_torch_fused",
            torch_compile=use_compile,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=2,
            eval_strategy="steps" if val_ds else "no",
            eval_steps=args.eval_steps if val_ds else None,
            load_best_model_at_end=val_ds is not None,
            metric_for_best_model="eval_loss" if val_ds else None,
            report_to=[] if args.no_wandb else ["wandb"],
            seed=args.seed,
        ),
    )

    # Auto-resume
    last_ckpt = None
    ckpts = sorted(Path(args.output_dir).glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if ckpts:
        last_ckpt = str(ckpts[-1])
        log.info(f"Resuming from {last_ckpt}")

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        log.info(f"GPU: {gpu.name}, {gpu.total_memory / 1024**3:.1f} GB")

    trainer.train(resume_from_checkpoint=last_ckpt)

    if torch.cuda.is_available():
        log.info(f"Peak VRAM: {torch.cuda.max_memory_reserved() / 1024**3:.1f} GB")

    # Save
    final = Path(args.output_dir) / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))

    if n_new > 0:
        with torch.no_grad():
            sid_emb = model.get_input_embeddings().weight[-n_new:].float().cpu().numpy()
        np.save(final / "sid_embeddings.npy", sid_emb)
    else:
        sid_emb = np.empty((0, model.config.hidden_size))

    (final / "training_meta.json").write_text(json.dumps({
        "stage": "vocab_expansion",
        "base_model": args.model_name,
        "vocab_size": len(tokenizer),
        "new_tokens": n_new,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "sid_embedding_shape": list(sid_emb.shape),
    }, indent=2))
    log.info(f"Saved to {final} (SID embeddings: {sid_emb.shape})")


if __name__ == "__main__":
    main()
