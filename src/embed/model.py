"""Загрузка модели и генерация эмбеддингов."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel

from .config import PoolingStrategy

if TYPE_CHECKING:
    from mipt_master.src.device import DeviceInfo

    from .config import EmbedConfig

logger = logging.getLogger("embed-items")


# =============================================================================
# Pooling strategies
# =============================================================================


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Last token pooling (Qwen, GTE, E5-Mistral).

    Используется в decoder-based моделях.
    - Если left-padding: берём последний токен
    - Если right-padding: берём последний не-padding токен

    Args:
        last_hidden_states: Hidden states ``[B, seq_len, hidden_dim]``.
        attention_mask: Маска внимания ``[B, seq_len]``.

    Returns:
        Эмбеддинги ``[B, hidden_dim]``.
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]

    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def mean_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean pooling (Jina, Stella, E5, Nomic).

    Усредняем hidden states по всем не-padding токенам.

    Args:
        last_hidden_states: Hidden states ``[B, seq_len, hidden_dim]``.
        attention_mask: Маска внимания ``[B, seq_len]``.

    Returns:
        Эмбеддинги ``[B, hidden_dim]``.
    """
    # Расширяем маску для broadcasting: [B, seq_len] -> [B, seq_len, 1]
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()

    # Суммируем только не-padding токены
    sum_embeddings = torch.sum(last_hidden_states * mask_expanded, dim=1)

    # Нормализуем на число токенов
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)

    return sum_embeddings / sum_mask


def cls_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """CLS token pooling (BGE, BERT-based, MixedBread).

    Берём первый токен ([CLS]).

    Args:
        last_hidden_states: Hidden states ``[B, seq_len, hidden_dim]``.
        attention_mask: Маска внимания ``[B, seq_len]`` (не используется).

    Returns:
        Эмбеддинги ``[B, hidden_dim]``.
    """
    return last_hidden_states[:, 0]


def get_pooling_fn(strategy: PoolingStrategy):
    """Получить функцию pooling по стратегии.

    Args:
        strategy: PoolingStrategy enum.

    Returns:
        Callable для pooling.
    """
    pooling_fns = {
        PoolingStrategy.LAST_TOKEN: last_token_pool,
        PoolingStrategy.MEAN: mean_pool,
        PoolingStrategy.CLS: cls_pool,
    }
    return pooling_fns[strategy]


# =============================================================================
# Model loading
# =============================================================================


def load_model(
    config: EmbedConfig,
    device_info: DeviceInfo,
) -> AutoModel:
    """Загрузить и подготовить модель для генерации эмбеддингов.

    Args:
        config: Конфигурация.
        device_info: Информация об устройстве.

    Returns:
        Модель, готовая к инференсу.
    """
    logger.info(f"Loading model: {config.model_name}")
    logger.info(f"Pooling strategy: {config.pooling_strategy.value}")

    # Загружаем модель
    model = AutoModel.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )

    # Перемещаем на устройство
    model = model.to(device_info.device)
    model.eval()

    # torch.compile только для CUDA
    if config.use_compile and device_info.supports_compile:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    logger.info(f"Model hidden size: {model.config.hidden_size}, target_dim: {config.target_dim}")

    return model


# =============================================================================
# Embedding generation
# =============================================================================


@torch.no_grad()
def generate_embeddings(
    model: AutoModel,
    batch: dict[str, Tensor],
    device: str,
    target_dim: int = 1024,
    pooling_strategy: PoolingStrategy = PoolingStrategy.LAST_TOKEN,
) -> np.ndarray:
    """Сгенерировать эмбеддинги для батча.

    Args:
        model: Модель для генерации.
        batch: Словарь с input_ids и attention_mask.
        device: Устройство для вычислений.
        target_dim: Целевая размерность (truncate если больше).
        pooling_strategy: Стратегия pooling.

    Returns:
        L2-нормализованные эмбеддинги ``[B, target_dim]``.
    """
    # Перемещаем на устройство
    encoded = {k: v.to(device) for k, v in batch.items()}

    # Forward pass
    outputs = model(**encoded)

    # Применяем pooling
    pooling_fn = get_pooling_fn(pooling_strategy)
    embeddings = pooling_fn(outputs.last_hidden_state, encoded["attention_mask"])

    # Truncate до target_dim если нужно (Matryoshka)
    if target_dim and target_dim < embeddings.shape[1]:
        embeddings = embeddings[:, :target_dim]

    # L2 normalize
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()


# =============================================================================
# Verification
# =============================================================================


def verify_embedding_consistency(
    model: AutoModel,
    tokenized_data: dict[str, np.ndarray],
    device: str,
    target_dim: int,
    batch_size: int,
    pooling_strategy: PoolingStrategy = PoolingStrategy.LAST_TOKEN,
) -> bool:
    """Проверить consistency: single vs batch embedding.

    Проверяет, что эмбеддинг одного элемента совпадает с эмбеддингом
    того же элемента в батче. Это валидирует корректность padding.

    Args:
        model: Модель.
        tokenized_data: Pre-tokenized данные.
        device: Устройство.
        target_dim: Целевая размерность.
        batch_size: Размер батча.
        pooling_strategy: Стратегия pooling.

    Returns:
        True если consistency OK, False иначе.
    """
    logger.info("Verifying embedding consistency...")

    n_items = min(batch_size, tokenized_data["n_items"])

    # Batch embedding
    batch = {
        "input_ids": torch.from_numpy(tokenized_data["input_ids"][:n_items]),
        "attention_mask": torch.from_numpy(tokenized_data["attention_mask"][:n_items]),
    }
    batch_embeddings = generate_embeddings(
        model, batch, device, target_dim, pooling_strategy
    )

    # Single embedding
    single = {
        "input_ids": torch.from_numpy(tokenized_data["input_ids"][:1]),
        "attention_mask": torch.from_numpy(tokenized_data["attention_mask"][:1]),
    }
    single_embedding = generate_embeddings(
        model, single, device, target_dim, pooling_strategy
    )

    # Сравнение
    are_similar = np.allclose(single_embedding[0], batch_embeddings[0], rtol=1e-5, atol=1e-5)
    diff = np.abs(single_embedding[0] - batch_embeddings[0])

    logger.info(f"Embeddings are similar: {are_similar}")
    logger.info(f"Max difference: {diff.max():.2e}")
    logger.info(f"Mean difference: {diff.mean():.2e}")

    if not are_similar:
        logger.warning("Embeddings differ more than expected!")
        logger.warning("This may indicate issues with padding or batch processing.")
    else:
        logger.info("✓ Embedding consistency verified")

    return are_similar
