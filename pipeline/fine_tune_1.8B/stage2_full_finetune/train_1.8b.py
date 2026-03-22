#!/usr/bin/env python3
"""Stage 2: Full fine-tuning Qwen3-1.8B with Semantic IDs.

Unsloth (FastLanguageModel) for model loading + vanilla Trainer for training.
Custom instruction masking, sequence packing, SID eval callback.

Aligned with 8B pipeline for experimental consistency.
Based on Eugene Yan's approach + OpenOneRec best practices.
"""

from unsloth import FastLanguageModel, is_bfloat16_supported  # isort: skip  # must be first

import argparse
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Optional

import itertools

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage2-1.8b")

_THINK_BLOCK = "<think>\n\n</think>\n\n"


# ---------------------------------------------------------------------------
# Instruction masking
# ---------------------------------------------------------------------------

def _get_masking_ids(tokenizer) -> tuple[list[int], int]:
    template_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    log.info(f"Response template IDs: {template_ids} | <|im_end|>: {im_end_id}")
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
# Sequence packing
# ---------------------------------------------------------------------------

def pack_sequences(dataset: Dataset, max_length: int, pad_token_id: int) -> Dataset:
    """Greedy bin-packing: concatenate short sequences into fixed-length chunks.

    With avg ~150 tokens and max_length=512, packs ~3 sequences per chunk,
    giving ~3x throughput increase.
    """
    all_ids = dataset["input_ids"]
    all_labels = dataset["labels"]

    packed_ids, packed_labels, packed_mask = [], [], []
    buf_ids: list[int] = []
    buf_labels: list[int] = []

    def _flush():
        if not buf_ids:
            return
        pad_len = max_length - len(buf_ids)
        packed_ids.append(buf_ids + [pad_token_id] * pad_len)
        packed_labels.append(buf_labels + [-100] * pad_len)
        packed_mask.append([1] * len(buf_ids) + [0] * pad_len)

    for ids, labs in zip(all_ids, all_labels):
        if len(ids) > max_length:
            ids = ids[:max_length]
            labs = labs[:max_length]
        if buf_ids and len(buf_ids) + len(ids) > max_length:
            _flush()
            buf_ids, buf_labels = [], []
        buf_ids.extend(ids)
        buf_labels.extend(labs)

    _flush()

    log.info(
        f"Packing: {len(all_ids):,} sequences -> {len(packed_ids):,} chunks "
        f"(x{len(all_ids) / max(len(packed_ids), 1):.1f} compression)"
    )
    return Dataset.from_dict({
        "input_ids": packed_ids,
        "labels": packed_labels,
        "attention_mask": packed_mask,
    })


# ---------------------------------------------------------------------------
# SID evaluation callback
# ---------------------------------------------------------------------------

_SID_PATTERN = re.compile(
    r"<\|sid_start\|>"
    r"<\|A(\d+)\|><\|B(\d+)\|><\|C(\d+)\|><\|D(\d+)\|>"
    r"<\|sid_end\|>"
)


def _parse_sid(text: str) -> Optional[tuple[str, str, str, str]]:
    m = _SID_PATTERN.search(text)
    return tuple(m.groups()) if m else None


class SIDEvalCallback(TrainerCallback):
    """Generates SIDs and computes hierarchical accuracy during training.

    Metrics (following Eugene Yan's RecommendationEvalCallback):
      - valid_format, level_A, level_AB, level_ABC, exact (Hit@1)
    """

    def __init__(self, tokenizer, val_path: str, eval_every: int,
                 n_samples: int = 200, max_new_tokens: int = 12, seed: int = 42):
        self.tokenizer = tokenizer
        self.eval_every = eval_every
        self.max_new_tokens = max_new_tokens
        self.eval_data: list[dict] = []

        ds = load_dataset("parquet", data_files=val_path, split="train")
        reco_types = {
            "copurchase_forward", "copurchase_backward",
            "seq_last_2", "seq_last_3", "seq_last_5",
        }
        reco_indices = [i for i, t in enumerate(ds["type"]) if t in reco_types]
        if not reco_indices:
            log.warning("SIDEvalCallback: no reco tasks in val data, disabling")
            return

        import random
        rng = random.Random(seed)
        chosen = rng.sample(reco_indices, min(n_samples, len(reco_indices)))

        for idx in chosen:
            conv = ds[idx]["conversations"]
            expected = conv[-1]["content"] if conv[-1]["role"] == "assistant" else None
            if not expected or not _parse_sid(expected):
                continue
            prompt_msgs = [m for m in conv if m["role"] != "assistant"]
            try:
                text = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True,
                )
            text = text.replace(_THINK_BLOCK, "")
            ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
            self.eval_data.append({"input_ids": ids, "expected": expected})

        log.info(f"SIDEvalCallback: {len(self.eval_data)} samples, eval every {eval_every} steps")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not self.eval_data or state.global_step == 0:
            return
        if state.global_step % self.eval_every != 0:
            return
        self._evaluate(model, state.global_step)

    @torch.no_grad()
    def _evaluate(self, model, step: int):
        was_training = model.training
        model.eval()

        counts = {"total": 0, "valid": 0, "A": 0, "AB": 0, "ABC": 0, "exact": 0}
        t0 = time.time()

        for item in self.eval_data:
            counts["total"] += 1
            input_ids = item["input_ids"].to(model.device)
            out = model.generate(input_ids=input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            generated = self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=False)
            pred = _parse_sid(generated)
            exp = _parse_sid(item["expected"])
            if not pred or not exp:
                continue
            counts["valid"] += 1
            if pred[0] == exp[0]: counts["A"] += 1
            if pred[:2] == exp[:2]: counts["AB"] += 1
            if pred[:3] == exp[:3]: counts["ABC"] += 1
            if pred == exp: counts["exact"] += 1

        n = counts["total"]
        if n > 0:
            log.info(
                f"SID Eval (step {step}, {time.time()-t0:.0f}s, n={n}): "
                f"valid={100*counts['valid']/n:.0f}% | "
                f"A={100*counts['A']/n:.0f}% | AB={100*counts['AB']/n:.0f}% | "
                f"ABC={100*counts['ABC']/n:.0f}% | exact={100*counts['exact']/n:.1f}%"
            )
        model.train(was_training)


# ---------------------------------------------------------------------------
# Model snapshot callback
# ---------------------------------------------------------------------------

class ModelSnapshotCallback(TrainerCallback):
    """Saves model + tokenizer (without optimizer) every N steps."""

    def __init__(self, snapshot_steps: int, output_dir: Path,
                 tokenizer, max_snapshots: int = 3):
        self.snapshot_steps = snapshot_steps
        self.snapshots_dir = output_dir / "snapshots"
        self.tokenizer = tokenizer
        self.max_snapshots = max_snapshots
        self.saved: list[Path] = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.snapshot_steps != 0 or state.global_step == 0:
            return
        snap_dir = self.snapshots_dir / f"step-{state.global_step}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(snap_dir)
        self.tokenizer.save_pretrained(snap_dir)
        self.saved.append(snap_dir)
        while len(self.saved) > self.max_snapshots:
            old = self.saved.pop(0)
            shutil.rmtree(old, ignore_errors=True)
        log.info(f"Snapshot: {snap_dir.name} ({len(self.saved)}/{self.max_snapshots})")


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


def load_reco_dataset(
    path: str, tokenizer, template_ids: list[int], im_end_id: int,
    max_samples: Optional[int], max_length: int, seed: int,
) -> Dataset:
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
# General data mixing (reasoning SFT datasets)
# ---------------------------------------------------------------------------

# Supported reasoning datasets and their field mappings
_REASONING_SOURCES = {
    "nvidia/OpenMathReasoning": {
        "split": "cot",           # chain-of-thought solutions
        "user_field": "problem",
        "assistant_field": "generated_solution",
    },
    "nvidia/OpenCodeReasoning": {
        "config": "split_0",       # HF requires explicit config name
        "split": "split_0",        # split name matches config name
        "user_field": "input",
        "assistant_field": "output",
    },
    "glaiveai/reasoning-v1-20m": {
        "split": "train",
        "user_field": "prompt",
        "assistant_field": "response",
    },
}

# Default proportions within the general mix (math-heavy, following OpenOneRec)
_DEFAULT_MIX_WEIGHTS = {
    "nvidia/OpenMathReasoning": 0.60,
    "nvidia/OpenCodeReasoning": 0.20,
    "glaiveai/reasoning-v1-20m": 0.20,
}

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning traces from responses."""
    return _THINK_RE.sub("", text).strip()


def _to_conversation(user_text: str, assistant_text: str) -> list[dict]:
    """Convert a Q&A pair to chat format compatible with apply_chat_template."""
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": _strip_think(assistant_text)},
    ]


def load_general_dataset(
    tokenizer, template_ids: list[int], im_end_id: int,
    max_length: int, n_samples: int, seed: int,
    sources: Optional[dict[str, float]] = None,
) -> Dataset:
    """Load reasoning SFT datasets, convert to conversations, tokenize with instruction masking.

    Sources: dict of {hf_dataset_name: weight}. Weights are relative (normalized to sum=1).
    Uses the same instruction masking as SID data for consistent loss computation.
    """
    sources = sources or _DEFAULT_MIX_WEIGHTS
    total_weight = sum(sources.values())

    all_rows = []
    for source_name, weight in sources.items():
        n = int(n_samples * weight / total_weight)
        if n == 0:
            continue

        cfg = _REASONING_SOURCES.get(source_name)
        if cfg is None:
            log.warning(f"Unknown source {source_name!r}, skipping")
            continue

        hf_config = cfg.get("config")
        log.info(f"Loading {source_name!r} (n={n:,}, split={cfg['split']}, config={hf_config})")
        load_kw = {"split": cfg["split"], "streaming": True}
        if hf_config:
            load_kw["name"] = hf_config
        ds = load_dataset(source_name, **load_kw)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)

        rows = []
        for ex in itertools.islice(ds, n):
            user = ex.get(cfg["user_field"], "")
            assistant = ex.get(cfg["assistant_field"], "")
            if user and assistant:
                rows.append({"conversations": _to_conversation(user, assistant)})

        log.info(f"  Collected {len(rows):,} examples from {source_name}")
        all_rows.extend(rows)

    if not all_rows:
        log.warning("No general data loaded")
        return Dataset.from_dict({"input_ids": [], "labels": [], "attention_mask": []})

    log.info(f"General data total: {len(all_rows):,} conversations")
    ds = Dataset.from_list(all_rows)

    def process(batch):
        texts = [_apply_template(tokenizer, conv) for conv in batch["conversations"]]
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        enc["labels"] = [
            _apply_chat_mask(ids, template_ids, im_end_id)
            for ids in enc["input_ids"]
        ]
        return enc

    ds = ds.map(process, batched=True, remove_columns=ds.column_names, num_proc=4,
                desc="Tokenizing general reasoning data")
    log.info(f"General dataset tokenized: {len(ds):,} examples")
    return ds


def mix_datasets(
    reco_ds: Dataset, general_fraction: float,
    tokenizer, template_ids: list[int], im_end_id: int,
    max_length: int, seed: int,
    sources: Optional[dict[str, float]] = None,
) -> Dataset:
    """Mix SID conversations with reasoning SFT data.

    Args:
        reco_ds: SID conversation dataset (already tokenized).
        general_fraction: Fraction of general data in total mix (e.g. 0.25).
            0 = no mixing. SID data count stays fixed, general is added on top.
        tokenizer: For tokenizing general data.
        template_ids, im_end_id: For instruction masking of general data.
        max_length: Max sequence length.
        seed: Random seed.
        sources: Dict of {hf_dataset: weight}. Default: math 60%, code 20%, reasoning 20%.
    """
    if general_fraction <= 0:
        return reco_ds

    n_reco = len(reco_ds)
    n_general = int(n_reco * general_fraction / (1.0 - general_fraction))

    general_ds = load_general_dataset(
        tokenizer, template_ids, im_end_id,
        max_length, n_general, seed, sources,
    )

    mixed = concatenate_datasets([reco_ds, general_ds]).shuffle(seed=seed)
    log.info(
        f"Data mix: {n_reco:,} SID ({100*n_reco/len(mixed):.0f}%) + "
        f"{len(general_ds):,} reasoning ({100*len(general_ds)/len(mixed):.0f}%) "
        f"= {len(mixed):,} total"
    )
    return mixed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Stage 2: Full fine-tuning (Qwen3-1.8B)")
    p.add_argument("--stage1-model", required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--val-file", default=None)
    p.add_argument("--output-dir", default="output")

    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=2_000)

    # Hyperparams (aligned with 8B for experimental consistency)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)

    # General data mixing (reasoning SFT datasets)
    p.add_argument("--general-fraction", type=float, default=0.0,
                   help="Fraction of general reasoning data in mix (0 = no mixing, 0.25 = 25%%)")

    p.add_argument("--packing", action="store_true")
    p.add_argument("--no-torch-compile", action="store_true")
    p.add_argument("--snapshot-steps", type=int, default=2000)
    p.add_argument("--max-snapshots", type=int, default=3)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--sid-eval-samples", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    eff_batch = args.batch_size * args.grad_accum
    log.info(f"Stage 2: full fine-tuning from {args.stage1_model}")
    log.info(f"  LR={args.lr}, warmup={args.warmup_ratio}, weight_decay={args.weight_decay}")
    log.info(f"  eff_batch={args.batch_size}x{args.grad_accum}={eff_batch}, epochs={args.epochs}")

    # --- Model (Unsloth) ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.stage1_model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        load_in_4bit=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify SID tokens
    sid = "<|sid_start|><|A10|><|B20|><|C30|><|D0|><|sid_end|>"
    ids = tokenizer.encode(sid, add_special_tokens=False)
    assert tokenizer.decode(ids, skip_special_tokens=False) == sid, "SID round-trip failed"
    log.info(f"  vocab={len(tokenizer):,}, SID round-trip OK")

    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True
    log.info(f"  params={sum(p.numel() for p in model.parameters()):,} (all trainable)")

    # --- Data ---
    template_ids, im_end_id = _get_masking_ids(tokenizer)

    train_ds = load_reco_dataset(
        args.train_file, tokenizer, template_ids, im_end_id,
        args.max_train_samples, args.max_seq_length, args.seed,
    )
    train_ds = mix_datasets(
        train_ds, args.general_fraction,
        tokenizer, template_ids, im_end_id,
        args.max_seq_length, args.seed,
    )
    if args.packing:
        train_ds = pack_sequences(train_ds, args.max_seq_length, tokenizer.pad_token_id)

    val_ds = (
        load_reco_dataset(
            args.val_file, tokenizer, template_ids, im_end_id,
            args.max_val_samples, args.max_seq_length, args.seed,
        )
        if args.val_file and Path(args.val_file).exists()
        else None
    )

    # --- Callbacks ---
    output_path = Path(args.output_dir)
    callbacks = [
        ModelSnapshotCallback(args.snapshot_steps, output_path, tokenizer, args.max_snapshots),
    ]
    val_for_eval = args.val_file or args.train_file
    if val_for_eval and Path(val_for_eval).exists():
        sid_cb = SIDEvalCallback(
            tokenizer, val_for_eval, eval_every=args.eval_steps,
            n_samples=args.sid_eval_samples, seed=args.seed,
        )
        if sid_cb.eval_data:
            callbacks.append(sid_cb)

    # --- Trainer ---
    use_compile = not args.no_torch_compile
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForCausalLM(tokenizer),
        callbacks=callbacks,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,

            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            lr_scheduler_type="cosine_with_min_lr",
            lr_scheduler_kwargs={"min_lr_rate": 0.2},
            max_grad_norm=1.0,
            optim="adamw_8bit",
            seed=args.seed,

            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            torch_compile=use_compile,
            torch_compile_backend="inductor" if use_compile else None,

            dataloader_num_workers=4,
            dataloader_pin_memory=True,

            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.snapshot_steps,
            save_total_limit=2,
            eval_strategy="steps" if val_ds else "no",
            eval_steps=args.eval_steps if val_ds else None,
            report_to=[] if args.no_wandb else ["wandb"],
        ),
    )

    # Auto-resume
    last_ckpt = None
    ckpts = sorted(output_path.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if ckpts:
        last_ckpt = str(ckpts[-1])
        log.info(f"Resuming from {last_ckpt}")

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        log.info(f"GPU: {gpu.name}, {gpu.total_memory / 1024**3:.1f} GB")

    result = trainer.train(resume_from_checkpoint=last_ckpt)

    if torch.cuda.is_available():
        log.info(f"Peak VRAM: {torch.cuda.max_memory_reserved() / 1024**3:.1f} GB")
    log.info(f"Loss: {result.metrics.get('train_loss', '?'):.4f}, "
             f"time: {result.metrics.get('train_runtime', 0) / 60:.1f} min")

    # --- Save ---
    final = output_path / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))

    (final / "training_meta.json").write_text(json.dumps({
        "stage1": args.stage1_model,
        "vocab_size": len(tokenizer),
        "epochs": args.epochs,
        "lr": args.lr,
        "eff_batch": eff_batch,
        "max_seq_length": args.max_seq_length,
        "train_samples": len(train_ds),
        "packing": args.packing,
        "general_fraction": args.general_fraction,
        "final_loss": result.metrics.get("train_loss"),
        "runtime_sec": result.metrics.get("train_runtime"),
    }, indent=2))
    log.info(f"Saved to {final}")


if __name__ == "__main__":
    main()
