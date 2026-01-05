"""Загрузка модели для full fine-tuning."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from .config import FullFineTuneConfig

logger = logging.getLogger("finetune-full")


# Токены для проверки после загрузки Stage 1
SEMANTIC_ID_TOKENS = [
    "<|rec|>",
    "<|sid_start|>",
    "<|sid_end|>",
    "<|sid_0|>",
    "<|sid_1023|>",
]


def verify_semantic_tokens(
    tokenizer: PreTrainedTokenizerBase,
    tokens: list[str] | None = None,
) -> bool:
    """Проверить наличие semantic ID токенов в токенизаторе.

    Args:
        tokenizer: Токенизатор.
        tokens: Список токенов для проверки (None = дефолт).

    Returns:
        True если все токены присутствуют.
    """
    if tokens is None:
        tokens = SEMANTIC_ID_TOKENS

    vocab = tokenizer.get_vocab()
    all_present = True

    for token in tokens:
        if token in vocab:
            logger.info(f"✓ Found token: {token}")
        else:
            logger.warning(f"⚠ Missing token: {token}")
            all_present = False

    return all_present


def test_sid_roundtrip(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Проверить round-trip кодирование/декодирование SID токенов.

    Args:
        tokenizer: Токенизатор.

    Returns:
        True если round-trip успешен.
    """
    logger.info("Testing SID round-trip...")

    sid_string = "<|rec|><|sid_start|><|sid_0|><|sid_256|><|sid_512|><|sid_768|><|sid_end|>"
    token_ids = tokenizer.encode(sid_string, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)

    logger.info(f"SID token IDs: {token_ids}")
    logger.info(f"Decoded SID: {decoded}")

    if decoded != sid_string:
        logger.error("❌ Round-trip SID mismatch — проблема со словарём!")
        return False

    logger.info("✓ SID round-trip successful")
    return True


def ensure_embedding_alignment(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Проверить и исправить alignment между моделью и токенизатором.

    Args:
        model: Модель.
        tokenizer: Токенизатор.
    """
    vocab_size = len(tokenizer)
    in_size = model.get_input_embeddings().weight.shape[0]
    out_size = model.get_output_embeddings().weight.shape[0]

    logger.info(f"Input emb size:  {in_size}")
    logger.info(f"Output emb size: {out_size}")
    logger.info(f"Tokenizer vocab: {vocab_size}")

    if out_size != vocab_size:
        logger.warning("Output head size != vocab size. Resizing...")
        model.resize_token_embeddings(vocab_size)
        out_size_new = model.get_output_embeddings().weight.shape[0]
        logger.info(f"After resize, output emb size: {out_size_new}")

        if out_size_new != vocab_size:
            logger.error("❌ Resize failed — возможны проблемы с генерацией.")
        else:
            logger.info("✓ Resize successful")
    else:
        logger.info("✓ Embedding & LM head sizes consistent with tokenizer")


def load_model_for_full_finetune(
    config: FullFineTuneConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Загрузить Stage 1 модель для full fine-tuning.

    Все float-параметры будут trainable.

    Args:
        config: Конфигурация.

    Returns:
        Кортеж (модель, токенизатор).
    """
    # Lazy import unsloth
    from unsloth import FastLanguageModel

    logger.info(f"Loading Stage 1 checkpoint: {config.stage1_checkpoint}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.stage1_checkpoint,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
    )

    logger.info(f"Loaded model with vocab size: {len(tokenizer):,}")

    # Проверяем semantic ID токены
    verify_semantic_tokens(tokenizer)

    # Проверяем round-trip
    if not test_sid_roundtrip(tokenizer):
        raise RuntimeError("SID round-trip failed — check vocabulary extension")

    # Gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing ENABLED")
    else:
        model.gradient_checkpointing_disable()
        logger.info("Gradient checkpointing DISABLED")

    # Делаем все float-параметры trainable
    trainable_params = 0
    total_params = 0
    float_dtypes = (torch.float16, torch.float32, torch.bfloat16)

    for name, p in model.named_parameters():
        total_params += p.numel()
        if p.dtype in float_dtypes:
            p.requires_grad = True
            trainable_params += p.numel()

    pct = 100 * trainable_params / total_params if total_params > 0 else 0
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({pct:.2f}%)")

    # Проверяем alignment эмбеддингов
    ensure_embedding_alignment(model, tokenizer)

    return model, tokenizer


def get_trainable_params_info(model: PreTrainedModel) -> dict:
    """Получить информацию о trainable параметрах.

    Args:
        model: Модель.

    Returns:
        Словарь с информацией.
    """
    trainable = 0
    total = 0

    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    return {
        "trainable": trainable,
        "total": total,
        "percentage": 100 * trainable / total if total > 0 else 0,
    }

