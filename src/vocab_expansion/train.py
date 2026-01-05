"""Основной тренировочный цикл для vocabulary expansion."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from .callbacks import create_callbacks
from .dataset import DataCollatorLM, prepare_datasets

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from .config import VocabExpansionConfig

logger = logging.getLogger("vocab-expansion")


@dataclass
class TrainResult:
    """Результат обучения.

    Attributes:
        global_step: Финальный глобальный шаг.
        training_loss: Финальный training loss.
        runtime_seconds: Время обучения в секундах.
        samples_per_second: Скорость обучения.
    """

    global_step: int
    training_loss: Optional[float]
    runtime_seconds: float
    samples_per_second: float


def _log_gpu_stats(start_memory: float, max_memory: float, runtime: float) -> None:
    """Залогировать статистику GPU после обучения.

    Args:
        start_memory: Память в начале (GB).
        max_memory: Максимальная память GPU (GB).
        runtime: Время обучения (секунды).
    """
    used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    used_for_training = round(used_memory - start_memory, 3)
    used_pct = round(used_memory / max_memory * 100, 3)
    training_pct = round(used_for_training / max_memory * 100, 3)

    logger.info(f"Training time: {runtime:.1f}s ({runtime / 60:.1f} min)")
    logger.info(f"Peak memory: {used_memory} GB ({used_pct}% of max)")
    logger.info(f"Memory for training: {used_for_training} GB ({training_pct}% of max)")


def _setup_wandb(config: VocabExpansionConfig) -> Optional[str]:
    """Инициализировать wandb если включён.

    Args:
        config: Конфигурация.

    Returns:
        Run name или None.
    """
    if not config.use_wandb:
        return None

    try:
        import wandb

        run_name = f"vocab-ext-{config.category}-lr{config.learning_rate}"
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
        return run_name
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        return None


def _log_to_wandb(data: dict, config: VocabExpansionConfig) -> None:
    """Логировать данные в wandb.

    Args:
        data: Данные для логирования.
        config: Конфигурация.
    """
    if not config.use_wandb:
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(data)
    except Exception:
        pass


def train_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: VocabExpansionConfig,
    num_new_tokens: int,
    test_prompts: list[list[dict]],
) -> TrainResult:
    """Основной цикл обучения эмбеддингов.

    Args:
        model: Подготовленная модель.
        tokenizer: Токенизатор.
        config: Конфигурация.
        num_new_tokens: Число добавленных токенов.
        test_prompts: Тестовые промпты для генерации.

    Returns:
        TrainResult с метриками обучения.
    """
    logger.info("Starting Stage 1: Embedding initialization training")

    # Подготовка данных
    train_dataset, val_dataset = prepare_datasets(config, tokenizer)

    _log_to_wandb(
        {
            "dataset/train_size": len(train_dataset),
            "dataset/val_size": len(val_dataset),
            "dataset/vocabulary_size": len(tokenizer),
            "dataset/new_tokens": num_new_tokens,
        },
        config,
    )

    # Data collator
    data_collator = DataCollatorLM(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps if config.max_steps > 0 else -1,
        num_train_epochs=config.num_train_epochs if config.max_steps <= 0 else 1,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.steps_per_train_log,
        save_steps=config.save_steps,
        save_total_limit=2,
        report_to=["wandb"] if config.use_wandb else [],
        optim=config.optim,
        seed=config.random_state,
        fp16=False,
        bf16=False,
    )

    # Callbacks
    callbacks, data_inspection_callback = create_callbacks(
        tokenizer=tokenizer,
        config=config,
        num_new_tokens=num_new_tokens,
        test_prompts=test_prompts,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    data_inspection_callback.set_trainer(trainer)

    # GPU stats
    start_gpu_memory = 0.0
    max_memory = 0.0
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        max_memory = round(gpu_stats.total_memory / 1024**3, 3)
        logger.info(f"GPU: {gpu_stats.name}, Max memory: {max_memory} GB")
        logger.info(f"Reserved at start: {start_gpu_memory} GB")

    # Train
    train_result = trainer.train()

    # Log GPU stats
    if torch.cuda.is_available():
        _log_gpu_stats(start_gpu_memory, max_memory, train_result.metrics["train_runtime"])

    # Финальные метрики
    training_loss = getattr(train_result, "training_loss", None)
    global_step = getattr(train_result, "global_step", config.max_steps)

    if config.use_wandb:
        try:
            import wandb

            if wandb.run is not None:
                wandb.summary["final_loss"] = training_loss
                wandb.summary["total_steps"] = global_step
                wandb.summary["training_time_seconds"] = train_result.metrics["train_runtime"]
        except Exception:
            pass

    logger.info("Stage 1 embedding initialization completed!")

    return TrainResult(
        global_step=global_step,
        training_loss=training_loss,
        runtime_seconds=train_result.metrics["train_runtime"],
        samples_per_second=train_result.metrics.get("train_samples_per_second", 0),
    )


def save_model_and_tokenizer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: VocabExpansionConfig,
    train_result: Optional[TrainResult] = None,
) -> Path:
    """Сохранить модель и токенизатор.

    Args:
        model: Обученная модель.
        tokenizer: Токенизатор.
        config: Конфигурация.
        train_result: Результат обучения (опционально).

    Returns:
        Путь к сохранённой модели.
    """
    logger.info("Saving model and tokenizer")

    # Проверка размеров
    input_size = model.get_input_embeddings().weight.shape[0]
    output_size = model.get_output_embeddings().weight.shape[0]
    vocab_size = len(tokenizer)

    logger.info("=== Pre-save verification ===")
    logger.info(f"Input embeddings: {input_size}")
    logger.info(f"Output embeddings: {output_size}")
    logger.info(f"Vocabulary: {vocab_size}")

    if input_size != vocab_size or output_size != vocab_size:
        logger.warning("Size mismatch — forcing resize")
        model.resize_token_embeddings(vocab_size)
        input_size = model.get_input_embeddings().weight.shape[0]
        output_size = model.get_output_embeddings().weight.shape[0]
        logger.info(f"After resize: Input={input_size}, Output={output_size}")

    assert input_size == vocab_size, f"Input mismatch: {input_size} != {vocab_size}"
    assert output_size == vocab_size, f"Output mismatch: {output_size} != {vocab_size}"
    logger.info("✓ All dimensions verified")

    # Сохраняем
    final_save_path = config.output_dir / "final"
    final_save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving to: {final_save_path}")
    model.save_pretrained(str(final_save_path))
    tokenizer.save_pretrained(str(final_save_path))

    # Сохраняем конфиг обучения
    config_dict = {
        "stage": "vocab_extension",
        "model_name": config.model_name,
        "num_semantic_tokens": config.num_semantic_tokens,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "category": config.category,
        "vocabulary_size": len(tokenizer),
    }
    if train_result:
        config_dict["final_loss"] = train_result.training_loss
        config_dict["global_step"] = train_result.global_step
        config_dict["runtime_seconds"] = train_result.runtime_seconds

    with open(final_save_path / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Model and tokenizer saved!")
    logger.info(f"Checkpoint: {final_save_path}")

    return final_save_path


def save_embeddings_artifact(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: VocabExpansionConfig,
    num_new_tokens: int,
    train_result: Optional[TrainResult] = None,
) -> Path:
    """Сохранить эмбеддинги как артефакт.

    Args:
        model: Обученная модель.
        tokenizer: Токенизатор.
        config: Конфигурация.
        num_new_tokens: Число новых токенов.
        train_result: Результат обучения.

    Returns:
        Путь к сохранённым эмбеддингам.
    """
    logger.info("Saving embeddings artifact")

    embeddings = model.get_input_embeddings().weight
    new_embeddings = embeddings[len(tokenizer) - num_new_tokens:].detach().cpu()

    embeddings_path = config.output_dir / "semantic_embeddings.npy"
    np.save(embeddings_path, new_embeddings.float().numpy())
    logger.info(f"Saved embeddings: {embeddings_path}")

    # Сохраняем в wandb если включён
    if config.use_wandb:
        try:
            import wandb

            if wandb.run is not None:
                artifact = wandb.Artifact(
                    f"semantic_embeddings_{config.category}",
                    type="embeddings",
                    description=f"Trained semantic ID embeddings for {config.category}",
                    metadata={
                        "num_tokens": num_new_tokens,
                        "model": config.model_name,
                        "steps": train_result.global_step if train_result else config.max_steps,
                        "final_loss": train_result.training_loss if train_result else None,
                    },
                )
                artifact.add_file(str(embeddings_path))
                wandb.log_artifact(artifact)
                logger.info("Logged embeddings to wandb")
        except Exception as e:
            logger.warning(f"Failed to log artifact to wandb: {e}")

    return embeddings_path

