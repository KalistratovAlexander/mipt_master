"""Конфигурация для inference и evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("inference")


def is_bf16_supported() -> bool:
    """Проверить поддержку bfloat16."""
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    return False


def get_device() -> str:
    """Определить доступное устройство."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class InferenceConfig:
    """Конфигурация для inference и evaluation.

    Attributes:
        model_path: Путь к базовой модели (Stage 1 или merged).
        lora_path: Путь к LoRA адаптеру (опционально).
        use_peft: Использовать PEFT для загрузки LoRA (vs unsloth).

        device: Устройство (cuda/mps/cpu, auto если None).
        dtype: Тип данных (auto если None).
        load_in_4bit: Загружать в 4-bit.
        load_in_8bit: Загружать в 8-bit.

        max_new_tokens: Максимум новых токенов.
        temperature: Температура семплирования.
        top_p: Top-p (nucleus) семплирование.
        top_k: Top-k семплирование.
        do_sample: Использовать семплирование (False = greedy).

        output_dir: Директория для результатов.
        save_results: Сохранять результаты в файл.
    """

    # Model paths
    model_path: str = "models/qwen3_fashion_vocab/final"
    lora_path: Optional[str] = None
    use_peft: bool = True  # True = pure PEFT, False = unsloth

    # Device settings
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # Generation settings
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("eval_results"))
    save_results: bool = True

    def __post_init__(self) -> None:
        """Автоопределение device и dtype."""
        if self.device is None:
            self.device = get_device()

        if self.dtype is None:
            if self.device == "cuda":
                self.dtype = torch.bfloat16 if is_bf16_supported() else torch.float16
            elif self.device == "mps":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32

        self.output_dir = Path(self.output_dir)
        if self.save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Проверить валидность конфигурации."""
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if self.lora_path:
            lora_path = Path(self.lora_path)
            if not lora_path.exists():
                raise FileNotFoundError(f"LoRA adapter not found: {self.lora_path}")

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")

    def log_config(self) -> None:
        """Вывести конфигурацию в лог."""
        logger.info("=== Inference Configuration ===")

        logger.info("Model:")
        logger.info(f"  model_path: {self.model_path}")
        logger.info(f"  lora_path: {self.lora_path or 'None'}")
        logger.info(f"  use_peft: {self.use_peft}")

        logger.info("Device:")
        logger.info(f"  device: {self.device}")
        logger.info(f"  dtype: {self.dtype}")
        logger.info(f"  load_in_4bit: {self.load_in_4bit}")
        logger.info(f"  load_in_8bit: {self.load_in_8bit}")

        logger.info("Generation:")
        logger.info(f"  max_new_tokens: {self.max_new_tokens}")
        logger.info(f"  temperature: {self.temperature}")
        logger.info(f"  top_p: {self.top_p}")
        logger.info(f"  top_k: {self.top_k}")
        logger.info(f"  do_sample: {self.do_sample}")

        logger.info("Output:")
        logger.info(f"  output_dir: {self.output_dir}")
        logger.info(f"  save_results: {self.save_results}")
        logger.info("===============================")

