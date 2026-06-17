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
import datasets
from datasets import load_dataset

datasets.disable_caching()
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
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
    """Initialize last n_new rows from MVN(mean, full_cov) of existing rows.

    OpenOneRec paper §4.2: Cholesky on full covariance with ridge regularization
    (1e-4 × mean diag) for numerical stability. FP32 throughout.
    """
    existing = weight[:-n_new].float()
    mean = existing.mean(dim=0)
    centered = existing - mean
    cov = (centered.T @ centered) / (existing.shape[0] - 1)
    ridge = 1e-4 * cov.diagonal().mean()
    cov = cov + ridge * torch.eye(cov.shape[0], device=cov.device, dtype=torch.float32)
    L = torch.linalg.cholesky(cov)
    z = torch.randn(n_new, weight.shape[1], device=weight.device, dtype=torch.float32)
    weight[-n_new:] = (mean + z @ L.T).to(weight.dtype)


def _init_h2(model, n_new: int, h2: dict) -> None:
    """H2 ablation init. Dispatches to pipeline/h2_init_ablation/init_strategies."""
    import sys
    sys.path.insert(0, h2["module_path"])
    from init_strategies import apply_init_to_model  # noqa: E402

    codebook = torch.load(h2["codebook_path"], map_location="cpu") if h2["arm"] == "D" else None
    titles = None
    if h2["arm"] == "C":
        with open(h2["title_map_path"]) as f:
            titles = json.load(f)

    apply_init_to_model(
        model=model,
        arm=h2["arm"],
        seed=h2["seed"],
        target_frobenius_ctrl=h2["target_frobenius_ctrl"],
        target_frobenius_sid=h2["target_frobenius_sid"],
        rqvae_codebook=codebook,
        title_token_ids_per_sid=titles,
    )
    log.info(
        f"H2 init: arm={h2['arm']} seed={h2['seed']} "
        f"target_ctrl={h2['target_frobenius_ctrl']:.6f} "
        f"target_sid={h2['target_frobenius_sid']:.6f}"
    )


def extend_vocabulary(model, tokenizer, h2: dict | None = None) -> int:
    tokens = make_sid_tokens()
    n_before = len(tokenizer)
    tokenizer.add_tokens(tokens, special_tokens=True)
    n_new = len(tokenizer) - n_before
    if n_new == 0:
        log.info("SID tokens already in vocab")
        return 0

    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        if h2 is not None:
            _init_h2(model, n_new, h2)
        else:
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
# Training monitor callbacks
# ---------------------------------------------------------------------------

class TrainingMonitorCallback(TrainerCallback):
    """NaN loss detection, grad_norm warning, per-log VRAM reporting."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("loss")
        grad_norm = logs.get("grad_norm")
        if loss is not None and loss != loss:
            raise RuntimeError(f"NaN loss at step {state.global_step} — aborting")
        if loss is not None and state.global_step <= 2 and loss < 0.01:
            log.warning(f"Loss={loss:.4f} at step {state.global_step} — verify label masking")
        if grad_norm is not None and grad_norm > 10.0:
            log.warning(f"grad_norm={grad_norm:.2f} at step {state.global_step} (>10 threshold)")
        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            log.info(f"[step {state.global_step}] VRAM={vram_gb:.1f} GB")


class EmbeddingMonitorCallback(TrainerCallback):
    """Tracks SID embedding norms vs existing tokens — catches collapse or frozen embeddings."""

    def __init__(self, model, n_new: int, log_every: int = 50):
        self.n_new = n_new
        self.log_every = log_every
        with torch.no_grad():
            W = model.get_input_embeddings().weight
            self._old_mean = W[:-n_new].float().norm(dim=-1).mean().item()
            self._init_new_mean = W[-n_new:].float().norm(dim=-1).mean().item()
        log.info(
            f"EmbeddingMonitor init: new_norm_mean={self._init_new_mean:.3f} "
            f"old_norm_mean={self._old_mean:.3f}"
        )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step == 0 or state.global_step % self.log_every != 0:
            return
        with torch.no_grad():
            W = model.get_input_embeddings().weight
            new_norms = W[-self.n_new:].float().norm(dim=-1)
            old_mean = W[:-self.n_new].float().norm(dim=-1).mean().item()
        log.info(
            f"[step {state.global_step}] emb_new: "
            f"mean={new_norms.mean():.3f} min={new_norms.min():.3f} max={new_norms.max():.3f} "
            f"| emb_old_mean={old_mean:.3f}"
        )


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

    # --- H2 ablation ----------------------------------------------------------
    # Default 'original' preserves the existing 1.7B training path byte-identical.
    p.add_argument("--init-strategy", choices=["original", "A", "B", "C", "D"],
                   default="original",
                   help="Embedding init for new SID tokens. 'original' = legacy diag-std; "
                        "A/B/C/D dispatch to pipeline/h2_init_ablation/init_strategies.")
    p.add_argument("--init-seed", type=int, default=None,
                   help="Seed for H2 init sampling. Defaults to --seed when omitted.")
    p.add_argument("--target-frobenius-ctrl", type=float, default=None,
                   help="Pre-registered Frobenius target for the 3 control-token rows "
                        "(required when --init-strategy != original).")
    p.add_argument("--target-frobenius-sid", type=float, default=None,
                   help="Pre-registered Frobenius target for the 1024 SID-token rows "
                        "(required when --init-strategy != original).")
    p.add_argument("--rqvae-codebook-path", default=None,
                   help="Path to RQ-VAE codebook .pt (required for arm D).")
    p.add_argument("--title-map-path", default=None,
                   help="Path to title_token_ids_per_sid.json (required for arm C).")
    p.add_argument("--h2-module-path", default=None,
                   help="Path to pipeline/h2_init_ablation dir (auto-detected from script location if omitted).")
    args = p.parse_args()

    h2 = None
    if args.init_strategy != "original":
        if args.target_frobenius_ctrl is None or args.target_frobenius_sid is None:
            p.error("--target-frobenius-ctrl and --target-frobenius-sid are required "
                    "when --init-strategy != original")
        module_path = args.h2_module_path or str(
            Path(__file__).resolve().parents[2] / "h2_init_ablation"
        )
        if not Path(module_path).exists():
            p.error(f"H2 module dir not found: {module_path}; pass --h2-module-path")
        h2 = {
            "arm": args.init_strategy,
            "seed": args.init_seed if args.init_seed is not None else args.seed,
            "target_frobenius_ctrl": args.target_frobenius_ctrl,
            "target_frobenius_sid": args.target_frobenius_sid,
            "module_path": module_path,
            "codebook_path": args.rqvae_codebook_path,
            "title_map_path": args.title_map_path,
        }

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

    try:
        import liger_kernel  # noqa: F401
        use_liger = True
        log.info("liger-kernel detected: fused kernels enabled")
    except ImportError:
        use_liger = False

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    n_new = extend_vocabulary(model, tokenizer, h2=h2)
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
    callbacks = [
        TrainingMonitorCallback(),
        EmbeddingMonitorCallback(model, n_new),
    ]
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
            lr_scheduler_type="cosine_with_min_lr",
            lr_scheduler_kwargs={"min_lr_rate": 0.1},
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            # weight_decay=0 for embeddings: L2 pulls ALL rows toward zero each step,
            # including tokens absent from batch — kills new SID embeddings before they learn
            weight_decay=0.0,
            bf16=True,
            tf32=True,
            use_liger_kernel=use_liger,
            gradient_checkpointing=False,
            adam_beta2=0.95,
            optim="adamw_torch_fused",
            torch_compile=use_compile,
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=4,
            dataloader_persistent_workers=True,
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
        "init_strategy": args.init_strategy,
        "init_seed": args.init_seed if args.init_seed is not None else args.seed,
        "target_frobenius_ctrl": args.target_frobenius_ctrl,
        "target_frobenius_sid": args.target_frobenius_sid,
    }, indent=2))
    log.info(f"Saved to {final} (SID embeddings: {sid_emb.shape})")


if __name__ == "__main__":
    main()
