"""Работа с данными для RQ-VAE.

Этот модуль содержит всю логику работы с данными:
- загрузка эмбеддингов из parquet
- разбиение на train/val
- создание DataLoader'ов

Пример использования
--------------------
>>> from rqvae.data import prepare_data
>>> train_loader, val_loader = prepare_data(
...     path="data/embeddings.parquet",
...     config=config,
...     device_info=device_info,
... )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from mipt_master.src.device import DeviceInfo

    from .config import RQVAEConfig

logger = logging.getLogger("train-rqvae")


def load_embeddings(path: str | Path) -> torch.Tensor:
    """Загрузить эмбеддинги из parquet.

    Читает parquet memory-efficient способом: через pyarrow буфер + reshape,
    без создания промежуточных Python-списков.

    Args:
        path: Путь к parquet-файлу с колонкой ``embedding``.

    Returns:
        Тензор формы ``[N, D]`` (например, ``[50768, 1024]``).

    Raises:
        FileNotFoundError: Файл не найден.
        ValueError: Файл пустой.

    Формат parquet (фиксированный)
    ------------------------------
    - Колонка: ``embedding``
    - Тип: ``list<float>`` или ``large_list<float>``
    - Все векторы одной длины (например, 1024)
    - Без null-значений
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    logger.info(f"Loading embeddings from {path}")

    table = pq.read_table(path, columns=["embedding"])
    if table.num_rows == 0:
        raise ValueError(f"Embeddings file is empty: {path}")

    # combine_chunks() возвращает Array (не ChunkedArray)
    emb_arr = table.column("embedding").combine_chunks()

    # Достаём плоский буфер + offsets → reshape [N, D]
    offsets = emb_arr.offsets.to_numpy(zero_copy_only=False)
    d = int(offsets[1] - offsets[0])
    flat = emb_arr.values.to_numpy(zero_copy_only=False)

    # Конвертируем в float32 и делаем writable для PyTorch
    if flat.dtype != np.float32:
        flat = flat.astype(np.float32)
    elif not flat.flags.writeable:
        flat = flat.copy()

    embeddings = torch.from_numpy(flat.reshape(-1, d)).contiguous()

    logger.info(f"Loaded {len(embeddings):,} embeddings, dim={d}")
    return embeddings


def build_loaders(
    embeddings: torch.Tensor,
    config: RQVAEConfig,
    device_info: DeviceInfo,
) -> tuple[DataLoader, DataLoader]:
    """Разбить эмбеддинги на train/val и создать DataLoader'ы.

    Args:
        embeddings: Тензор эмбеддингов ``[N, D]``.
        config: Конфигурация RQ-VAE (нужны: val_split, seed, batch_size).
        device_info: Информация об устройстве (для num_workers, pin_memory и т.д.).

    Returns:
        Кортеж ``(train_loader, val_loader)``.
    """
    n = len(embeddings)
    val_size = int(n * config.val_split)
    train_size = n - val_size

    # Tensor поддерживает Dataset-интерфейс (__len__, __getitem__)
    train_dataset, val_dataset = torch.utils.data.random_split(
        embeddings,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    logger.info(f"Train size: {train_size:,}, Val size: {val_size:,}")

    # Используем оптимальные параметры DataLoader для устройства
    loader_kwargs = device_info.get_dataloader_kwargs()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )

    # Для val_loader отключаем persistent_workers на MPS
    val_kwargs = device_info.get_dataloader_kwargs()
    if device_info.is_mps:
        val_kwargs = {"num_workers": 0, "pin_memory": False}

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        **val_kwargs,
    )

    return train_loader, val_loader


def prepare_data(
    path: str | Path,
    config: RQVAEConfig,
    device_info: DeviceInfo,
) -> tuple[DataLoader, DataLoader]:
    """Загрузить данные и подготовить DataLoader'ы (удобная обёртка).

    Объединяет ``load_embeddings`` + ``build_loaders`` в один вызов.

    Args:
        path: Путь к parquet-файлу с эмбеддингами.
        config: Конфигурация RQ-VAE.
        device_info: Информация об устройстве.

    Returns:
        Кортеж ``(train_loader, val_loader)``.

    Example:
        >>> train_loader, val_loader = prepare_data(
        ...     "data/embeddings.parquet", config, device_info
        ... )
    """
    embeddings = load_embeddings(path)
    return build_loaders(embeddings, config, device_info)
