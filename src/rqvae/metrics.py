"""Метрики для оценки качества RQ-VAE.

Функции для вычисления метрик во время обучения и валидации:
- unique_ids_proportion: доля уникальных semantic ID в батче
- codebook_usage: использование кодбука по уровням
- avg_residual_norm: средняя L2-норма остатка после квантования
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from torch import nn


def unique_ids_proportion(semantic_ids: Tensor) -> float:
    """Доля уникальных semantic ID в батче.

    Semantic ID считается уникальным, если нет другого ID в батче
    с таким же набором кодов по всем уровням.

    Args:
        semantic_ids: Тензор ``[B, L]`` — индексы кодов для каждого уровня.

    Returns:
        Доля уникальных ID (от 0 до 1).

    Example:
        >>> ids = torch.tensor([[0, 1, 2], [0, 1, 2], [1, 2, 3]])
        >>> unique_ids_proportion(ids)
        0.333...  # только [1, 2, 3] уникален
    """
    batch_size = semantic_ids.shape[0]
    if batch_size <= 1:
        return 1.0

    # Сравниваем все пары: [B, 1, L] == [1, B, L] → [B, B, L]
    ids_expanded_1 = semantic_ids.unsqueeze(1)
    ids_expanded_2 = semantic_ids.unsqueeze(0)
    matches = (ids_expanded_1 == ids_expanded_2).all(dim=-1)  # [B, B]

    # Верхний треугольник (без диагонали) — дубликаты
    upper_tri_matches = torch.triu(matches, diagonal=1)
    has_duplicate = upper_tri_matches.any(dim=1)

    n_unique = (~has_duplicate).sum().item()
    return n_unique / batch_size


def codebook_usage(vq_layers: nn.ModuleList) -> list[float]:
    """Использование кодбука по уровням.

    Возвращает долю кодов, которые были использованы хотя бы раз
    с момента последнего сброса счётчика.

    Args:
        vq_layers: Список VQ-слоёв модели (``model.vq_layers``).

    Returns:
        Список usage rate для каждого уровня (от 0 до 1).

    Example:
        >>> usage = codebook_usage(model.vq_layers)
        >>> print(usage)  # [0.95, 0.87, 0.72]
    """
    return [vq_layer.get_usage_rate() for vq_layer in vq_layers]


def avg_residual_norm(residual: Tensor) -> float:
    """Средняя L2-норма остатка после квантования.

    Показывает, сколько информации "потеряно" после всех уровней
    квантования. Чем меньше — тем лучше реконструкция.

    Args:
        residual: Тензор остатка ``[B, D]`` из ``loss_dict["residual"]``.

    Returns:
        Средняя L2-норма по батчу.

    Example:
        >>> res = torch.randn(64, 32)
        >>> avg_residual_norm(res)
        5.6...
    """
    return residual.norm(dim=-1).mean().item()

