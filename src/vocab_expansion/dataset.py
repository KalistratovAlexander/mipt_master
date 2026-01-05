"""Загрузка и токенизация датасета с semantic ID диалогами."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from datasets import Dataset, load_dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .config import VocabExpansionConfig

logger = logging.getLogger("vocab-expansion")


def load_conversation_dataset(
    config: VocabExpansionConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: str = "train",
) -> Dataset:
    """Загрузить parquet с диалогами и применить chat template.

    Args:
        config: Конфигурация.
        tokenizer: Токенизатор с chat template.
        split: train или val.

    Returns:
        Dataset с колонкой 'text' (применённый chat template).
    """
    logger.info(f"Loading conversation dataset ({split})")

    data_path = config.train_path if split == "train" else config.val_path

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading from: {data_path}")
    dataset = load_dataset("parquet", data_files=str(data_path), split="train")
    logger.info(f"Loaded {len(dataset)} conversations")

    # Определяем число примеров
    if split == "train":
        num_samples = min(len(dataset), config.max_training_samples)
    else:
        num_samples = min(len(dataset), config.val_samples)

    logger.info(f"Sampling {num_samples} examples for {split}")
    dataset = dataset.shuffle(seed=config.random_state).select(range(num_samples))

    # Применяем chat template
    def apply_chat_template(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=config.enable_thinking,
        )
        return {"text": text}

    logger.info("Applying chat template...")
    dataset = dataset.map(
        apply_chat_template,
        remove_columns=dataset.column_names,
        num_proc=config.num_proc,
    )

    logger.info(f"Created dataset with {len(dataset)} examples")

    # Проверяем наличие semantic ID токенов
    if split == "train" and len(dataset) > 0:
        sample_text = dataset[0]["text"]
        if "<|sid_start|>" in sample_text and "<|sid_end|>" in sample_text:
            logger.info("✓ Semantic ID tokens found in dataset")
            sid_count = sample_text.count("<|sid_start|>")
            logger.info(f"  Sample contains {sid_count} SID(s)")
            logger.info("=" * 60)
            logger.info(f"Sample ({split}):\n{sample_text[:500]}...")
            logger.info("=" * 60)
        else:
            logger.warning("⚠ No semantic ID tokens in sample")

    return dataset


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    config: VocabExpansionConfig,
) -> Dataset:
    """Токенизировать датасет для language modeling.

    Args:
        dataset: Dataset с колонкой 'text'.
        tokenizer: Токенизатор.
        config: Конфигурация.

    Returns:
        Dataset с колонками input_ids, attention_mask, labels.
    """
    def tokenize_batch(batch: dict) -> dict:
        out = tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=config.max_seq_length,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    logger.info("Tokenizing dataset...")
    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        num_proc=config.num_proc,
    )
    return dataset


class DataCollatorLM:
    """Data collator для language modeling с динамическим padding.

    Маскирует pad токены в labels значением -100 для игнорирования в loss.

    Attributes:
        tokenizer: Токенизатор с pad_token_id.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Инициализация.

        Args:
            tokenizer: Токенизатор.
        """
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        """Собрать батч с динамическим padding.

        Args:
            features: Список примеров с input_ids и attention_mask.

        Returns:
            Батч с input_ids, attention_mask, labels.
        """
        import torch

        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        # Маскируем pad токены в labels
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch


def prepare_datasets(
    config: VocabExpansionConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[Dataset, Dataset]:
    """Подготовить train и val датасеты.

    Args:
        config: Конфигурация.
        tokenizer: Токенизатор.

    Returns:
        Кортеж (train_dataset, val_dataset).
    """
    # Загружаем сырые данные
    raw_train = load_conversation_dataset(config, tokenizer, split="train")
    raw_val = load_conversation_dataset(config, tokenizer, split="val")

    # Токенизируем
    train_dataset = tokenize_dataset(raw_train, tokenizer, config)
    val_dataset = tokenize_dataset(raw_val, tokenizer, config)

    logger.info(f"Train dataset: {len(train_dataset)} examples")
    logger.info(f"Val dataset: {len(val_dataset)} examples")

    return train_dataset, val_dataset

