"""Функции для расширения токенизатора semantic ID токенами."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from .config import VocabExpansionConfig

logger = logging.getLogger("vocab-expansion")


def _ensure_model_vocab_alignment(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    context: str = "",
) -> None:
    """Проверить и исправить alignment между моделью и токенизатором.

    Args:
        model: Модель.
        tokenizer: Токенизатор.
        context: Контекст для логирования.
    """
    vocab_size = len(tokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    lm_head_size = model.get_output_embeddings().weight.shape[0]

    if embedding_size == vocab_size == lm_head_size:
        return  # All aligned

    if context:
        logger.warning(f"{context}: size mismatch detected")
    logger.warning(
        f"Vocab: {vocab_size:,}, Embeddings: {embedding_size:,}, LM head: {lm_head_size:,}"
    )

    model.resize_token_embeddings(vocab_size)

    new_embedding_size = model.get_input_embeddings().weight.shape[0]
    new_lm_head_size = model.get_output_embeddings().weight.shape[0]
    logger.info(f"After resize: Embeddings={new_embedding_size:,}, LM head={new_lm_head_size:,}")


def extend_tokenizer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: VocabExpansionConfig,
) -> int:
    """Добавить semantic ID токены в токенизатор.

    Добавляет:
    - <|rec|> — маркер рекомендации
    - <|sid_start|> — начало semantic ID
    - <|sid_end|> — конец semantic ID
    - <|sid_0|> ... <|sid_N|> — сами semantic ID токены

    Args:
        model: Модель для расширения.
        tokenizer: Токенизатор для расширения.
        config: Конфигурация.

    Returns:
        Число добавленных токенов.
    """
    # Lazy import unsloth (может не быть установлен)
    from unsloth import add_new_tokens

    logger.info("=== Extending tokenizer with semantic ID tokens ===")

    original_vocab_size = len(tokenizer)
    original_embedding_size = model.get_input_embeddings().weight.shape[0]
    original_lm_head_size = model.get_output_embeddings().weight.shape[0]

    logger.info(
        f"Before — Vocab: {original_vocab_size:,}, Embeddings: {original_embedding_size:,}, "
        f"LM head: {original_lm_head_size:,}"
    )

    # Выравниваем если нужно
    if original_embedding_size > original_vocab_size:
        logger.warning(
            f"Model has {original_embedding_size - original_vocab_size} more embeddings than vocab"
        )
        _ensure_model_vocab_alignment(model, tokenizer, "Pre-extension")

    # Формируем список новых токенов
    new_tokens = ["<|rec|>", "<|sid_start|>", "<|sid_end|>"]
    new_tokens.extend(f"<|sid_{i}|>" for i in range(config.num_semantic_tokens))

    logger.info(f"Adding {len(new_tokens)} new tokens:")
    logger.info("  Special: <|rec|>, <|sid_start|>, <|sid_end|>")
    logger.info(f"  Semantic IDs: <|sid_0|> ... <|sid_{config.num_semantic_tokens - 1}|>")

    # Добавляем через unsloth
    add_new_tokens(model, tokenizer, new_tokens=new_tokens)

    # Проверяем результат
    new_vocab_size = len(tokenizer)
    new_embedding_size = model.get_input_embeddings().weight.shape[0]
    new_lm_head_size = model.get_output_embeddings().weight.shape[0]

    logger.info(
        f"After — Vocab: {new_vocab_size:,}, Embeddings: {new_embedding_size:,}, "
        f"LM head: {new_lm_head_size:,}"
    )

    # Финальное выравнивание
    _ensure_model_vocab_alignment(model, tokenizer, "Post-extension")

    num_added = new_vocab_size - original_vocab_size
    logger.info(f"✓ Successfully added {num_added} new tokens")
    logger.info("=" * 50)

    return num_added


def prepare_model_for_embedding_training(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: VocabExpansionConfig,
    num_new_tokens: int,
) -> PreTrainedModel:
    """Подготовить модель для обучения только эмбеддингов.

    Замораживает все параметры кроме input/output embeddings.
    Включает gradient checkpointing если указано в конфиге.

    Args:
        model: Модель.
        tokenizer: Токенизатор.
        config: Конфигурация.
        num_new_tokens: Число добавленных токенов.

    Returns:
        Подготовленная модель.
    """
    logger.info("=== Preparing model for embedding-only training ===")

    current_vocab_size = len(tokenizer)
    current_embedding_size = model.get_input_embeddings().weight.shape[0]

    logger.info(
        f"Vocab: {current_vocab_size:,}, Embeddings: {current_embedding_size:,}, "
        f"New tokens: {num_new_tokens}"
    )

    if current_embedding_size != current_vocab_size:
        raise RuntimeError(
            f"Embedding size mismatch: {current_embedding_size} != {current_vocab_size}"
        )

    # Замораживаем все параметры
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем эмбеддинги
    embedding_layer = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    embedding_layer.weight.requires_grad = True
    if output_embeddings is not None:
        output_embeddings.weight.requires_grad = True
        logger.info("✓ Unfroze input & output embeddings")
    else:
        logger.warning("No output embeddings — only input embeddings will be trained")

    # Статистика trainable параметров
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = trainable_params / total_params * 100
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({pct:.2f}%)")

    # Логируем статистику новых эмбеддингов
    original_vocab_size = len(tokenizer) - num_new_tokens
    with torch.no_grad():
        new_embeddings = embedding_layer.weight[original_vocab_size:]
        logger.info("New embeddings statistics:")
        logger.info(f"  Shape: {new_embeddings.shape}")
        logger.info(f"  Mean: {new_embeddings.mean().item():.6f}")
        logger.info(f"  Std:  {new_embeddings.std().item():.6f}")
        logger.info(f"  Min:  {new_embeddings.min().item():.6f}")
        logger.info(f"  Max:  {new_embeddings.max().item():.6f}")

    # Gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    else:
        model.gradient_checkpointing_disable()
        logger.info("Gradient checkpointing disabled")

    model.config.use_cache = not config.gradient_checkpointing

    logger.info("=== Model preparation complete ===")
    return model

