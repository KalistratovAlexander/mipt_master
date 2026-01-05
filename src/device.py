"""Управление устройством для training и inference.

Предоставляет информацию об устройстве (CUDA/MPS/CPU) и оптимальные
параметры для DataLoader в зависимости от платформы.

Example:
    >>> from mipt_master.src.device import get_device_info
    >>> device_info = get_device_info()
    >>> print(device_info.device)  # "cuda", "mps", or "cpu"
    >>> model.to(device_info.device)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DeviceInfo:
    """Информация об устройстве и его возможностях.

    Неизменяемый dataclass с предвычисленными параметрами для
    оптимальной работы на разных платформах.

    Attributes:
        device: Строка устройства ("cuda", "mps", "cpu").
        is_cuda: True если NVIDIA GPU.
        is_mps: True если Apple Silicon GPU.
        is_cpu: True если CPU.
        supports_pin_memory: Поддержка pinned memory для DataLoader.
        supports_compile: Поддержка torch.compile.
        default_num_workers: Рекомендуемое число workers для DataLoader.
        default_prefetch_factor: Рекомендуемый prefetch_factor.
        supports_bf16: Поддержка bfloat16.
    """

    device: str
    is_cuda: bool
    is_mps: bool
    is_cpu: bool
    supports_pin_memory: bool
    supports_compile: bool
    default_num_workers: int
    default_prefetch_factor: Optional[int]
    supports_bf16: bool

    def get_dataloader_kwargs(
        self,
        num_workers: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
    ) -> dict:
        """Получить оптимальные kwargs для DataLoader.

        Args:
            num_workers: Переопределить число workers (None = default).
            prefetch_factor: Переопределить prefetch (None = default).

        Returns:
            Словарь kwargs для DataLoader.

        Example:
            >>> loader = DataLoader(dataset, **device_info.get_dataloader_kwargs())
        """
        workers = num_workers if num_workers is not None else self.default_num_workers
        prefetch = prefetch_factor if prefetch_factor is not None else self.default_prefetch_factor

        kwargs = {
            "num_workers": workers,
            "pin_memory": self.supports_pin_memory,
        }

        # prefetch_factor требует num_workers > 0
        if workers > 0 and prefetch is not None:
            kwargs["prefetch_factor"] = prefetch
            kwargs["persistent_workers"] = True

        return kwargs

    def get_dtype(self, prefer_bf16: bool = True) -> torch.dtype:
        """Получить оптимальный dtype для устройства.

        Args:
            prefer_bf16: Предпочитать bfloat16 если поддерживается.

        Returns:
            torch.dtype для использования.
        """
        if prefer_bf16 and self.supports_bf16:
            return torch.bfloat16
        elif self.is_cuda or self.is_mps:
            return torch.float16
        return torch.float32


def get_device_info(configure_precision: bool = True) -> DeviceInfo:
    """Определить устройство и его возможности.

    Автоматически определяет лучшее доступное устройство и возвращает
    DeviceInfo с оптимальными параметрами.

    Args:
        configure_precision: Настроить TF32 для CUDA (default True).

    Returns:
        DeviceInfo с информацией об устройстве.

    Example:
        >>> device_info = get_device_info()
        >>> model = model.to(device_info.device)
        >>> loader = DataLoader(ds, **device_info.get_dataloader_kwargs())
    """
    if torch.cuda.is_available():
        # Включаем TF32 для Ampere+ GPU (ускорение matmul)
        if configure_precision:
            torch.set_float32_matmul_precision("high")

        # Проверяем поддержку bf16
        supports_bf16 = torch.cuda.is_bf16_supported()

        return DeviceInfo(
            device="cuda",
            is_cuda=True,
            is_mps=False,
            is_cpu=False,
            supports_pin_memory=True,
            supports_compile=True,
            default_num_workers=16,
            default_prefetch_factor=8,
            supports_bf16=supports_bf16,
        )

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS (Apple Silicon) не поддерживает multiprocessing в DataLoader
        return DeviceInfo(
            device="mps",
            is_cuda=False,
            is_mps=True,
            is_cpu=False,
            supports_pin_memory=False,
            supports_compile=False,  # torch.compile на MPS пока экспериментальный
            default_num_workers=0,  # MPS не работает с num_workers > 0
            default_prefetch_factor=None,
            supports_bf16=False,  # MPS не поддерживает bf16
        )

    # CPU fallback
    return DeviceInfo(
        device="cpu",
        is_cuda=False,
        is_mps=False,
        is_cpu=True,
        supports_pin_memory=False,
        supports_compile=False,
        default_num_workers=4,
        default_prefetch_factor=2,
        supports_bf16=False,
    )


# Alias для обратной совместимости
DeviceManager = DeviceInfo


def setup_device() -> DeviceInfo:
    """Alias для get_device_info() для обратной совместимости."""
    return get_device_info()

