"""Dataset для pre-tokenized данных."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from mipt_master.src.device import DeviceInfo

    from .config import EmbedConfig

logger = logging.getLogger("embed-items")


class TokenizedDataset(Dataset):
    """Dataset для pre-tokenized данных с pinned memory support.

    Attributes:
        input_ids: Тензор input_ids ``[N, seq_len]``.
        attention_mask: Тензор attention_mask ``[N, seq_len]``.
        length: Число элементов.
    """

    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> None:
        """Инициализация с numpy arrays, конвертация в pinned tensors.

        Args:
            input_ids: Токены ``[N, seq_len]``.
            attention_mask: Маска внимания ``[N, seq_len]``.
        """
        # Конвертируем в тензоры и pin memory для быстрого GPU transfer
        self.input_ids = torch.from_numpy(input_ids).pin_memory()
        self.attention_mask = torch.from_numpy(attention_mask).pin_memory()
        self.length = len(input_ids)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def load_tokenized_data(path: Path) -> dict[str, np.ndarray]:
    """Загрузить pre-tokenized данные из .npz файла.

    Args:
        path: Путь к .npz файлу.

    Returns:
        Словарь с ключами: input_ids, attention_mask, n_items.

    Raises:
        FileNotFoundError: Если файл не найден.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Pre-tokenized data not found: {path}. Please run tokenize_items.py first."
        )

    logger.info(f"Loading pre-tokenized data from {path}")
    data = np.load(path)
    logger.info(f"Loaded: shape {data['input_ids'].shape}, n_items={data['n_items']}")

    return {
        "input_ids": data["input_ids"],
        "attention_mask": data["attention_mask"],
        "n_items": int(data["n_items"]),
    }


def build_dataloader(
    tokenized_data: dict[str, np.ndarray],
    config: EmbedConfig,
    device_info: DeviceInfo,
) -> DataLoader:
    """Создать DataLoader для генерации эмбеддингов.

    Args:
        tokenized_data: Словарь с input_ids, attention_mask.
        config: Конфигурация.
        device_info: Информация об устройстве.

    Returns:
        DataLoader для батчевой генерации эмбеддингов.
    """
    dataset = TokenizedDataset(
        tokenized_data["input_ids"],
        tokenized_data["attention_mask"],
    )

    # Получаем оптимальные параметры для устройства
    num_workers = device_info.default_num_workers
    persistent_workers = num_workers > 0
    prefetch_factor = device_info.default_prefetch_factor if num_workers > 0 else None

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,  # Сохраняем порядок
        num_workers=num_workers,
        pin_memory=False,  # Уже pinned в dataset
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    logger.info(
        f"DataLoader created: batch_size={config.batch_size}, "
        f"num_workers={num_workers}, total_batches={len(dataloader)}"
    )

    return dataloader

