"""Основной тренировочный цикл для full fine-tuning."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer

from .callbacks import create_callbacks
from .model import load_model_for_full_finetune

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from .config import FullFineTuneConfig

logger = logging.getLogger("finetune-full")


@dataclass
class TrainResult:
    """Результат обучения.

    Attributes:
        global_step: Финальный глобальный шаг.
        training_loss: Финальный training loss.
        runtime_seconds: Время обучения в секундах.
    """

    global_step: int
    training_loss: Optional[float]
    runtime_seconds: float


def get_latest_checkpoint(output_dir: Path) -> Optional[str]:
    """Найти последний чекпоинт в директории.

    Args:
        output_dir: Директория с чекпоинтами.

    Returns:
        Путь к последнему чекпоинту или None.
    """
    pattern = output_dir / "checkpoint-*"
    checkpoint_dirs = list(output_dir.glob("checkpoint-*"))

    if not checkpoint_dirs:
        logger.info("No existing checkpoints found")
        return None

    try:
        latest = max(
            checkpoint_dirs,
            key=lambda x: int(x.name.split("-")[-1])
        )
        logger.info(f"Found latest checkpoint: {latest}")
        return str(latest)
    except Exception as e:
        logger.warning(f"Error parsing checkpoint dirs: {e}")
        return None


def load_conversation_dataset(
    config: FullFineTuneConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: str = "train",
) -> Optional[Dataset]:
    """Загрузить parquet с диалогами и применить chat template.

    Args:
        config: Конфигурация.
        tokenizer: Токенизатор.
        split: train или val.

    Returns:
        Dataset с колонкой 'text' или None.
    """
    if split == "train":
        data_path = config.train_path
    elif split == "val":
        data_path = config.val_path
    else:
        raise ValueError(f"Unknown split: {split}")

    if not data_path or not data_path.exists():
        logger.warning(f"{split} dataset not found: {data_path}")
        return None

    logger.info(f"Loading {split} dataset from {data_path}")
    ds = load_dataset("parquet", data_files=str(data_path), split="train")
    logger.info(f"Loaded {len(ds):,} {split} examples")

    # Семплирование
    if split == "train":
        if not config.use_full_dataset and config.max_training_samples:
            n = min(len(ds), config.max_training_samples)
            logger.info(f"Sampling {n:,} train examples")
            ds = ds.shuffle(seed=config.random_state).select(range(n))
    else:
        n = min(len(ds), config.eval_samples)
        logger.info(f"Sampling {n:,} validation examples")
        ds = ds.shuffle(seed=config.random_state).select(range(n))

    # Применяем chat template
    def apply_template(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=config.enable_thinking,
        )
        return {"text": text}

    logger.info(f"Applying chat template for {split}...")
    ds = ds.map(
        apply_template,
        remove_columns=ds.column_names,
        num_proc=config.num_proc,
        batch_size=1000,
        writer_batch_size=5000,
    )

    logger.info(f"Prepared {split} dataset: {len(ds):,} rows")

    # Логируем пример
    if len(ds) > 0:
        sample = ds[0]["text"]
        token_ids = tokenizer.encode(sample, add_special_tokens=False)
        logger.info("=" * 60)
        logger.info(f"Sample ({split}): {sample[:500]}...")
        logger.info(f"Token count: {len(token_ids)}")
        logger.info("=" * 60)

    return ds


def _setup_wandb(config: FullFineTuneConfig) -> None:
    """Инициализировать wandb если включён.

    Args:
        config: Конфигурация.
    """
    if not config.use_wandb:
        return

    try:
        import wandb

        run_name = f"full-{config.category}-lr{config.learning_rate}-ep{config.num_train_epochs}"
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
                k: str(v) if isinstance(v, Path) else v
                for k, v in config.__dict__.items()
                if not k.startswith("_")
            },
            mode=config.wandb_mode,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        config.use_wandb = False


def _log_wandb_summary(result: TrainResult, config: FullFineTuneConfig) -> None:
    """Логировать summary в wandb.

    Args:
        result: Результат обучения.
        config: Конфигурация.
    """
    if not config.use_wandb:
        return

    try:
        import wandb

        if wandb.run is not None:
            wandb.summary["final_loss"] = result.training_loss
            wandb.summary["total_steps"] = result.global_step
            wandb.summary["train_runtime"] = result.runtime_seconds
    except Exception:
        pass


def _is_bf16_supported() -> bool:
    """Проверить поддержку bfloat16."""
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    return False


def finetune_model(config: FullFineTuneConfig) -> TrainResult:
    """Основной цикл full fine-tuning.

    Args:
        config: Конфигурация.

    Returns:
        TrainResult с метриками обучения.
    """
    logger.info("Starting Stage 2: Full fine-tuning")

    # Инициализируем wandb
    _setup_wandb(config)

    # Загружаем модель
    model, tokenizer = load_model_for_full_finetune(config)

    # Загружаем данные
    train_dataset = load_conversation_dataset(config, tokenizer, split="train")
    val_dataset = load_conversation_dataset(config, tokenizer, split="val")

    if train_dataset is None or len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty or not loaded")

    # Вычисляем warmup steps
    eff_batch = config.batch_size * config.gradient_accumulation_steps
    total_steps = len(train_dataset) * config.num_train_epochs // eff_batch
    warmup_steps = int(total_steps * config.warmup_ratio)

    logger.info(f"Total steps: {total_steps:,}, warmup: {warmup_steps:,}")

    # SFT Config
    bf16_supported = _is_bf16_supported()
    sft_config = SFTConfig(
        dataset_text_field="text",
        dataset_num_proc=config.num_proc,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.gradient_clip_norm,
        seed=config.random_state,
        output_dir=str(config.output_dir),
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=not bf16_supported,
        bf16=bf16_supported,
        report_to=["wandb"] if config.use_wandb else [],
        packing=False,
        max_seq_length=config.max_seq_length,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps if val_dataset is not None else None,
        metric_for_best_model=config.metric_for_best_model if val_dataset is not None else None,
        greater_is_better=config.greater_is_better,
        load_best_model_at_end=config.load_best_model_at_end and val_dataset is not None,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Callbacks
    callbacks = create_callbacks(config)

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        callbacks=callbacks,
    )

    # GPU stats
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name}, total memory: {props.total_memory / 1024**3:.2f} GB")

    # Resume from checkpoint
    resume_checkpoint = None
    if config.resume_from_checkpoint:
        resume_checkpoint = get_latest_checkpoint(config.output_dir)
        if resume_checkpoint:
            logger.info(f"Resuming from: {resume_checkpoint}")

    # Train
    logger.info("Starting training...")
    train_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
    logger.info("Training finished!")

    # GPU memory stats
    if torch.cuda.is_available():
        used = torch.cuda.max_memory_reserved() / 1024**3
        logger.info(f"Peak GPU memory: {used:.2f} GB")

    logger.info(f"Train stats: {train_stats.metrics}")

    result = TrainResult(
        global_step=train_stats.global_step,
        training_loss=train_stats.metrics.get("train_loss"),
        runtime_seconds=train_stats.metrics.get("train_runtime", 0),
    )

    _log_wandb_summary(result, config)

    return result


def save_final_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: FullFineTuneConfig,
    result: Optional[TrainResult] = None,
) -> Path:
    """Сохранить финальную модель.

    Args:
        model: Обученная модель.
        tokenizer: Токенизатор.
        config: Конфигурация.
        result: Результат обучения (опционально).

    Returns:
        Путь к сохранённой модели.
    """
    logger.info("Saving final model...")

    final_dir = config.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Сохраняем конфиг
    cfg = {
        "stage": "full_finetuning",
        "base_checkpoint": config.stage1_checkpoint,
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
        "category": config.category,
        "vocabulary_size": len(tokenizer),
        "use_full_dataset": config.use_full_dataset,
    }
    if result:
        cfg["final_loss"] = result.training_loss
        cfg["global_step"] = result.global_step
        cfg["runtime_seconds"] = result.runtime_seconds

    with open(final_dir / "training_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    logger.info(f"Model saved to: {final_dir}")
    return final_dir


def finish_wandb() -> None:
    """Завершить wandb сессию."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

