"""Основной тренировочный цикл для LoRA fine-tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer

from .callbacks import create_callbacks
from .model import load_model_with_lora

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from .config import LoRAConfig

logger = logging.getLogger("finetune-lora")


@dataclass
class TrainResult:
    """Результат обучения LoRA.

    Attributes:
        global_step: Финальный глобальный шаг.
        training_loss: Финальный training loss.
        runtime_seconds: Время обучения в секундах.
    """

    global_step: int
    training_loss: Optional[float]
    runtime_seconds: float


def load_conversation_dataset(
    config: LoRAConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: str = "train",
) -> Optional[Dataset]:
    """Загрузить parquet с диалогами и применить chat template.

    Args:
        config: Конфигурация.
        tokenizer: Токенизатор.
        split: train или val.

    Returns:
        Dataset с колонкой 'text' или None если не найден.
    """
    if split == "train":
        data_path = config.train_path
    elif split == "val":
        data_path = config.val_path
    else:
        raise ValueError(f"Unknown split: {split}")

    if not data_path.exists():
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
    )

    # Проверка наличия SID
    if len(ds) > 0:
        sample_text = ds[0]["text"]
        sid_count = sample_text.count("<|sid_start|>")
        logger.info(f"{split} sample: {len(sample_text)} chars, {sid_count} SIDs")
        logger.info("=" * 40)
        logger.info(sample_text[:500])
        logger.info("=" * 40)

    return ds


def _setup_wandb(config: LoRAConfig) -> None:
    """Инициализировать wandb если включён.

    Args:
        config: Конфигурация.
    """
    if not config.use_wandb:
        return

    try:
        import wandb

        run_name = f"lora-{config.category}-r{config.lora_r}-lr{config.learning_rate}"
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


def _log_wandb_summary(result: TrainResult, config: LoRAConfig) -> None:
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
            wandb.summary["train_loss"] = result.training_loss
            wandb.summary["train_runtime"] = result.runtime_seconds
            wandb.summary["train_steps"] = result.global_step
    except Exception:
        pass


def train_lora(
    config: LoRAConfig,
    generic_prompts: Optional[list[dict]] = None,
    sid_prompts: Optional[list[dict]] = None,
) -> TrainResult:
    """Основной цикл LoRA fine-tuning.

    Args:
        config: Конфигурация.
        generic_prompts: Generic тестовые промпты.
        sid_prompts: SID тестовые промпты.

    Returns:
        TrainResult с метриками обучения.
    """
    logger.info("Starting Stage 2: LoRA fine-tuning")

    # Инициализируем wandb
    _setup_wandb(config)

    # Загружаем модель с LoRA
    model, tokenizer = load_model_with_lora(config)

    # Загружаем данные
    train_dataset = load_conversation_dataset(config, tokenizer, split="train")
    val_dataset = load_conversation_dataset(config, tokenizer, split="val")

    if train_dataset is None or len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty or not loaded")

    # Вычисляем warmup steps
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    total_steps_est = len(train_dataset) * config.num_train_epochs // effective_batch
    warmup_steps = int(total_steps_est * config.warmup_ratio)

    logger.info(f"Estimated total steps: {total_steps_est:,}")
    logger.info(f"Warmup steps: {warmup_steps:,}")

    # SFT Config
    sft_config = SFTConfig(
        dataset_text_field="text",
        dataset_num_proc=config.num_proc,
        max_seq_length=config.max_seq_length,
        packing=False,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.gradient_clip_norm,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=config.eval_steps if val_dataset is not None else None,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        seed=config.random_state,
        fp16=(config.dtype == torch.float16),
        bf16=(config.dtype == torch.bfloat16),
        report_to=["wandb"] if config.use_wandb else [],
        output_dir=str(config.output_dir),
    )

    # Callbacks
    callbacks = create_callbacks(
        tokenizer=tokenizer,
        config=config,
        generic_prompts=generic_prompts,
        sid_prompts=sid_prompts,
    )

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
        gpu = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {gpu.name}, total memory: {gpu.total_memory / 1e9:.2f} GB")

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Training finished!")

    result = TrainResult(
        global_step=train_result.global_step,
        training_loss=getattr(train_result, "training_loss", None),
        runtime_seconds=train_result.metrics.get("train_runtime", 0),
    )

    _log_wandb_summary(result, config)

    return result


def save_lora_adapter(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: LoRAConfig,
) -> Path:
    """Сохранить LoRA адаптер.

    Args:
        model: Модель с LoRA.
        tokenizer: Токенизатор.
        config: Конфигурация.

    Returns:
        Путь к сохранённому адаптеру.
    """
    adapter_dir = config.output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving LoRA adapter to: {adapter_dir}")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    return adapter_dir


def save_merged_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: LoRAConfig,
) -> Optional[Path]:
    """Сохранить merged модель (LoRA + base).

    Args:
        model: Модель с LoRA.
        tokenizer: Токенизатор.
        config: Конфигурация.

    Returns:
        Путь к merged модели или None если не удалось.
    """
    if not config.save_merged:
        return None

    from unsloth import FastLanguageModel

    merged_dir = config.output_dir / "merged_full_model"
    merged_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged model to: {merged_dir}")

    try:
        FastLanguageModel.save_pretrained_merged(
            model,
            tokenizer,
            str(merged_dir),
            save_method="merged_16bit",
        )
        logger.info("Merged model saved successfully")
        return merged_dir
    except Exception as e:
        logger.warning(f"Could not save merged model: {e}")
        return None


def finish_wandb() -> None:
    """Завершить wandb сессию."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

