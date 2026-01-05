"""Загрузка модели и применение LoRA адаптеров."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from .config import LoRAConfig

logger = logging.getLogger("finetune-lora")


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
            logger.info(f"✓ token present: {token}")
        else:
            logger.warning(f"⚠ token missing: {token}")
            all_present = False

    return all_present


def load_model_with_lora(
    config: LoRAConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Загрузить Stage 1 модель и применить LoRA адаптеры.

    Args:
        config: Конфигурация LoRA.

    Returns:
        Кортеж (модель с LoRA, токенизатор).
    """
    # Lazy import unsloth
    from unsloth import FastLanguageModel

    logger.info(f"Loading Stage 1 checkpoint: {config.stage1_checkpoint}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.stage1_checkpoint,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )

    logger.info(f"Loaded tokenizer vocab size: {len(tokenizer):,}")

    # Проверяем semantic ID токены
    verify_semantic_tokens(tokenizer)

    # Применяем LoRA
    logger.info("Applying LoRA adapters...")
    logger.info(f"  r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    logger.info(f"  target_modules: {config.lora_target_modules}")
    logger.info(f"  use_rslora: {config.use_rslora}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth" if config.gradient_checkpointing else False,
        random_state=config.random_state,
        use_rslora=config.use_rslora,
        loftq_config=None,
    )

    # Логируем статистику параметров
    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    pct = 100 * trainable / total if total > 0 else 0
    logger.info(f"Trainable params (LoRA): {trainable:,} / {total:,} ({pct:.4f}%)")

    return model, tokenizer


def get_trainable_params_info(model: PreTrainedModel) -> dict:
    """Получить информацию о trainable параметрах.

    Args:
        model: Модель.

    Returns:
        Словарь с информацией о параметрах.
    """
    trainable = 0
    total = 0
    trainable_names = []

    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            trainable_names.append(name)

    return {
        "trainable": trainable,
        "total": total,
        "percentage": 100 * trainable / total if total > 0 else 0,
        "trainable_names": trainable_names,
    }

