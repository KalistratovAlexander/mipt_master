"""Загрузка моделей для inference."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from .config import InferenceConfig

logger = logging.getLogger("inference")


# Токены для проверки
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
    """Проверить наличие semantic ID токенов.

    Args:
        tokenizer: Токенизатор.
        tokens: Список токенов (None = дефолт).

    Returns:
        True если все токены присутствуют.
    """
    if tokens is None:
        tokens = SEMANTIC_ID_TOKENS

    vocab = tokenizer.get_vocab()
    all_present = True

    for token in tokens:
        if token in vocab:
            logger.debug(f"✓ Found: {token}")
        else:
            logger.warning(f"⚠ Missing: {token}")
            all_present = False

    return all_present


def load_model_peft(
    config: InferenceConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Загрузить модель через HuggingFace + PEFT.

    Args:
        config: Конфигурация.

    Returns:
        Кортеж (модель, токенизатор).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading tokenizer from: {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    logger.info(f"Vocab size: {len(tokenizer):,}")

    # Загружаем базовую модель
    logger.info(f"Loading base model from: {config.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=config.dtype,
        device_map=config.device,
        trust_remote_code=True,
    )

    # Применяем LoRA если указан
    if config.lora_path:
        from peft import PeftModel

        logger.info(f"Loading LoRA adapter from: {config.lora_path}")
        model = PeftModel.from_pretrained(model, config.lora_path)
        logger.info("✓ LoRA adapter loaded")

    model.eval()
    logger.info("Model ready for inference")

    return model, tokenizer


def load_model_unsloth(
    config: InferenceConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Загрузить модель через unsloth.

    Args:
        config: Конфигурация.

    Returns:
        Кортеж (модель, токенизатор).
    """
    from unsloth import FastLanguageModel

    # Определяем путь: если есть LoRA — грузим его, иначе базовую
    model_path = config.lora_path if config.lora_path else config.model_path

    logger.info(f"Loading model via unsloth: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Vocab size: {len(tokenizer):,}")

    # Для инференса переключаем в eval mode
    FastLanguageModel.for_inference(model)
    logger.info("Model ready for inference (unsloth)")

    return model, tokenizer


def load_model(
    config: InferenceConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Загрузить модель (автовыбор метода).

    Args:
        config: Конфигурация.

    Returns:
        Кортеж (модель, токенизатор).
    """
    if config.use_peft:
        model, tokenizer = load_model_peft(config)
    else:
        model, tokenizer = load_model_unsloth(config)

    # Проверяем semantic ID токены
    verify_semantic_tokens(tokenizer)

    return model, tokenizer


def get_model_info(model: PreTrainedModel) -> dict:
    """Получить информацию о модели.

    Args:
        model: Модель.

    Returns:
        Словарь с информацией.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
    }

