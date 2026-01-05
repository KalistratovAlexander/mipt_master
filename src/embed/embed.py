"""Основная логика генерации эмбеддингов."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from tqdm import tqdm

from .dataset import build_dataloader, load_tokenized_data
from .model import generate_embeddings, load_model, verify_embedding_consistency

if TYPE_CHECKING:
    from mipt_master.src.device import DeviceInfo

    from .config import EmbedConfig

logger = logging.getLogger("embed-items")


@dataclass
class EmbedResult:
    """Результат генерации эмбеддингов.

    Attributes:
        embeddings: Массив эмбеддингов ``[N, dim]``.
        total_items: Общее число элементов.
        total_time: Время обработки в секундах.
        items_per_sec: Скорость обработки.
    """

    embeddings: np.ndarray
    total_items: int
    total_time: float
    items_per_sec: float


def embed_items(
    config: EmbedConfig,
    device_info: DeviceInfo,
) -> EmbedResult:
    """Сгенерировать эмбеддинги для всех элементов.

    Args:
        config: Конфигурация.
        device_info: Информация об устройстве.

    Returns:
        EmbedResult с эмбеддингами и статистикой.
    """
    device = device_info.device

    # Загружаем pre-tokenized данные
    tokenized_data = load_tokenized_data(config.tokenized_path)
    total_items = tokenized_data["n_items"]

    # Загружаем модель
    model = load_model(config, device_info)

    # Проверяем consistency если включено
    if config.verify_consistency:
        verify_embedding_consistency(
            model,
            tokenized_data,
            device,
            config.target_dim,
            config.batch_size,
            config.pooling_strategy,
        )

    # Создаём DataLoader
    dataloader = build_dataloader(tokenized_data, config, device_info)

    # Pre-allocate output array
    logger.info(f"Pre-allocating output: {total_items:,} x {config.target_dim}")
    all_embeddings = np.zeros((total_items, config.target_dim), dtype=np.float32)

    # Генерируем эмбеддинги
    start_time = time.time()
    current_idx = 0

    with tqdm(total=total_items, desc="Generating embeddings") as pbar:
        for batch_idx, batch in enumerate(dataloader):
            try:
                batch_size = batch["input_ids"].size(0)

                # Генерируем эмбеддинги с правильной pooling стратегией
                batch_embeddings = generate_embeddings(
                    model,
                    batch,
                    device,
                    config.target_dim,
                    config.pooling_strategy,
                )

                # Записываем в pre-allocated array
                all_embeddings[current_idx : current_idx + batch_size] = batch_embeddings
                current_idx += batch_size

                pbar.update(batch_size)

                # Логируем прогресс
                if current_idx % config.log_freq == 0 or current_idx == total_items:
                    elapsed = time.time() - start_time
                    items_per_sec = current_idx / elapsed
                    eta = (total_items - current_idx) / items_per_sec if current_idx < total_items else 0
                    logger.info(
                        f"Processed {current_idx:,}/{total_items:,} "
                        f"({items_per_sec:.1f} items/sec, ETA: {eta / 60:.1f} min)"
                    )

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                raise

    total_time = time.time() - start_time
    items_per_sec = total_items / total_time

    logger.info("Embedding generation complete!")
    logger.info(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    logger.info(f"Average: {total_time / total_items * 1000:.1f} ms/item")

    # Проверяем нормализацию
    norms = np.linalg.norm(all_embeddings, axis=1)
    logger.info(f"L2 norms — Mean: {norms.mean():.6f}, Std: {norms.std():.6f}")

    return EmbedResult(
        embeddings=all_embeddings,
        total_items=total_items,
        total_time=total_time,
        items_per_sec=items_per_sec,
    )


def save_embeddings(
    config: EmbedConfig,
    result: EmbedResult,
    device_info: DeviceInfo,
) -> None:
    """Сохранить эмбеддинги в parquet-файл.

    Читает исходный parquet, добавляет колонку embedding, сохраняет.

    Args:
        config: Конфигурация.
        result: Результат генерации эмбеддингов.
        device_info: Информация об устройстве.
    """
    # Загружаем исходный DataFrame
    logger.info(f"Loading source data from {config.input_path}")
    item_df = pl.read_parquet(config.input_path)

    # Применяем ограничение если было
    if config.num_rows is not None:
        item_df = item_df.head(config.num_rows)

    # Проверяем соответствие
    if len(item_df) != result.total_items:
        raise ValueError(
            f"DataFrame has {len(item_df)} rows, but embeddings have {result.total_items}"
        )

    # Добавляем эмбеддинги
    embeddings_list = result.embeddings.tolist()
    item_df_with_embeddings = item_df.with_columns(
        pl.Series("embedding", embeddings_list, dtype=pl.List(pl.Float32))
    )

    # Сохраняем
    logger.info(f"Saving to {config.output_path}")
    item_df_with_embeddings.write_parquet(config.output_path)

    # Статистика
    file_size_mb = config.output_path.stat().st_size / 1024 / 1024
    logger.info("Final statistics:")
    logger.info(f"  Total items: {result.total_items:,}")
    logger.info(f"  Embedding dim: {result.embeddings.shape[1]}")
    logger.info(f"  File size: {file_size_mb:.1f} MB")
    logger.info(f"  Processing rate: {result.items_per_sec:.1f} items/sec")

    if device_info.is_cuda:
        import torch
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"  Peak GPU memory: {peak_memory_gb:.1f} GB")
