"""Конфигурация для Stage 1: Embedding initialization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("vocab-expansion")


def is_bf16_supported() -> bool:
    """Проверить поддержку bfloat16."""
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    return False


@dataclass
class VocabExpansionConfig:
    """Конфигурация расширения словаря и инициализации эмбеддингов.

    Stage 1: Добавляем semantic ID токены и обучаем только эмбеддинги.

    Attributes:
        model_name: Имя базовой модели (HuggingFace или unsloth).
        max_seq_length: Максимальная длина последовательности.
        dtype: Тип данных для весов (auto если None).
        load_in_4bit: Загружать модель в 4-bit.
        load_in_8bit: Загружать модель в 8-bit.
        random_state: Seed для воспроизводимости.
        num_proc: Число процессов для обработки данных.
        enable_thinking: Включить thinking mode (Qwen3).

        extend_vocabulary: Расширять ли словарь.
        codebook_levels: Число уровней иерархии (для RQ-VAE).
        codebook_size: Размер кодбука на каждом уровне.
        num_semantic_tokens: Число semantic ID токенов.

        category: Категория товаров.
        data_dir: Директория с данными.
        max_training_samples: Максимум примеров для обучения.
        val_samples: Число примеров для валидации.

        learning_rate: Learning rate.
        batch_size: Размер батча.
        gradient_accumulation_steps: Шаги аккумуляции градиента.
        max_steps: Максимум шагов (0 = по эпохам).
        num_train_epochs: Число эпох (игнорируется если max_steps > 0).
        warmup_steps: Шаги warmup.
        weight_decay: Weight decay.
        lr_scheduler_type: Тип scheduler.
        gradient_checkpointing: Использовать gradient checkpointing.
        optim: Оптимизатор.

        output_dir: Директория для сохранения.
        steps_per_train_log: Частота логов обучения.
        steps_per_val_log: Частота валидации.
        save_steps: Частота сохранения чекпоинтов.

        use_wandb: Использовать wandb для логирования.
        wandb_project: Имя проекта в wandb.
        wandb_mode: Режим wandb (online, offline, disabled).
    """

    # Model settings
    model_name: str = "unsloth/Qwen3-1.7B"
    max_seq_length: int = 1024
    dtype: Optional[torch.dtype] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    random_state: int = 1368
    num_proc: int = 8
    enable_thinking: bool = False

    # Semantic ID vocabulary extension
    extend_vocabulary: bool = True
    codebook_levels: int = 4
    codebook_size: int = 256
    num_semantic_tokens: int = 1024  # codebook_levels * codebook_size

    # Data settings
    category: str = "Amazon_Fashion"
    data_dir: Path = field(default_factory=lambda: Path("data"))
    max_training_samples: int = 16000
    val_samples: int = 500

    # Training params
    learning_rate: float = 1e-3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int = 1000
    num_train_epochs: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    optim: str = "adamw_8bit"

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("models/qwen3_fashion_vocab"))
    steps_per_train_log: int = 100
    steps_per_val_log: int = 250
    save_steps: int = 5000

    # Logging
    use_wandb: bool = True
    wandb_project: str = "semantic-id-vocab-extension"
    wandb_mode: str = "offline"  # online, offline, disabled

    # Computed paths (set in __post_init__)
    train_path: Optional[Path] = None
    val_path: Optional[Path] = None

    def __post_init__(self) -> None:
        """Автоопределение dtype и путей."""
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)

        if self.dtype is None:
            self.dtype = torch.bfloat16 if is_bf16_supported() else torch.float16

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_path = self.data_dir / "output" / f"{self.category}_conversations_train.parquet"
        self.val_path = self.data_dir / "output" / f"{self.category}_conversations_val.parquet"

    def validate(self) -> None:
        """Проверить валидность конфигурации."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        if self.codebook_levels <= 0:
            raise ValueError("codebook_levels must be > 0")
        if self.codebook_size <= 0:
            raise ValueError("codebook_size must be > 0")
        if self.num_semantic_tokens <= 0:
            raise ValueError("num_semantic_tokens must be > 0")

        expected_tokens = self.codebook_levels * self.codebook_size
        if self.num_semantic_tokens != expected_tokens:
            logger.warning(
                f"num_semantic_tokens ({self.num_semantic_tokens}) != "
                f"codebook_levels * codebook_size ({expected_tokens})"
            )

        if not self.train_path.exists():
            raise FileNotFoundError(
                f"Training data not found: {self.train_path}. "
                f"Run data preparation for category '{self.category}'."
            )

        if not self.val_path.exists():
            logger.warning(f"Validation data not found: {self.val_path}")

    def log_config(self) -> None:
        """Вывести конфигурацию в лог."""
        logger.info("=== Vocabulary Expansion Configuration ===")
        logger.info("Stage 1: Embedding Initialization")

        logger.info("Model Settings:")
        logger.info(f"  model_name: {self.model_name}")
        logger.info(f"  max_seq_length: {self.max_seq_length}")
        logger.info(f"  dtype: {self.dtype}")
        logger.info(f"  load_in_4bit: {self.load_in_4bit}")
        logger.info(f"  gradient_checkpointing: {self.gradient_checkpointing}")
        logger.info(f"  random_state: {self.random_state}")

        logger.info("Vocabulary Extension:")
        logger.info(f"  extend_vocabulary: {self.extend_vocabulary}")
        logger.info(f"  codebook_levels: {self.codebook_levels}")
        logger.info(f"  codebook_size: {self.codebook_size}")
        logger.info(f"  num_semantic_tokens: {self.num_semantic_tokens}")
        total_new = self.num_semantic_tokens + 3
        logger.info(f"  Total new tokens: {total_new} (incl. <|rec|>, <|sid_start|>, <|sid_end|>)")

        logger.info("Data Settings:")
        logger.info(f"  category: {self.category}")
        logger.info(f"  data_dir: {self.data_dir}")
        logger.info(f"  train_path: {self.train_path}")
        logger.info(f"  val_path: {self.val_path}")
        logger.info(f"  max_training_samples: {self.max_training_samples}")
        logger.info(f"  val_samples: {self.val_samples}")

        logger.info("Training Parameters:")
        logger.info(f"  learning_rate: {self.learning_rate}")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        logger.info(f"  effective_batch_size: {self.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"  max_steps: {self.max_steps}")
        logger.info(f"  num_train_epochs: {self.num_train_epochs}")
        logger.info(f"  warmup_steps: {self.warmup_steps}")
        logger.info(f"  weight_decay: {self.weight_decay}")
        logger.info(f"  lr_scheduler_type: {self.lr_scheduler_type}")
        logger.info(f"  optim: {self.optim}")

        logger.info("Output Settings:")
        logger.info(f"  output_dir: {self.output_dir}")
        logger.info(f"  steps_per_train_log: {self.steps_per_train_log}")
        logger.info(f"  steps_per_val_log: {self.steps_per_val_log}")
        logger.info(f"  save_steps: {self.save_steps}")

        logger.info("Logging:")
        logger.info(f"  use_wandb: {self.use_wandb}")
        logger.info(f"  wandb_project: {self.wandb_project}")
        logger.info(f"  wandb_mode: {self.wandb_mode}")
        logger.info("==========================================")

