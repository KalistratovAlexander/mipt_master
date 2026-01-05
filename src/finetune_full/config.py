"""Конфигурация для Stage 2: Full fine-tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("finetune-full")


def is_bf16_supported() -> bool:
    """Проверить поддержку bfloat16."""
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    return False


@dataclass
class FullFineTuneConfig:
    """Конфигурация full fine-tuning.

    Stage 2: Дообучаем всю модель (не LoRA) на диалогах с semantic IDs.

    Attributes:
        stage1_checkpoint: Путь к Stage 1 чекпоинту с расширенным словарём.
        max_seq_length: Максимальная длина последовательности.
        dtype: Тип данных для весов (auto если None).
        load_in_4bit: Загружать модель в 4-bit.
        load_in_8bit: Загружать модель в 8-bit.
        random_state: Seed для воспроизводимости.
        num_proc: Число процессов для обработки данных.
        enable_thinking: Включить thinking mode (Qwen3).

        category: Категория товаров.
        data_dir: Директория с данными.
        use_full_dataset: Использовать полный датасет.
        max_training_samples: Максимум примеров (если не full).
        eval_samples: Число примеров для валидации.

        learning_rate: Learning rate.
        batch_size: Размер батча.
        gradient_accumulation_steps: Шаги аккумуляции.
        gradient_clip_norm: Gradient clipping.
        max_steps: Максимум шагов (-1 = по эпохам).
        num_train_epochs: Число эпох.
        warmup_ratio: Доля warmup.
        weight_decay: Weight decay.
        lr_scheduler_type: Тип scheduler.

        gradient_checkpointing: Gradient checkpointing.
        optim: Оптимизатор.

        output_dir: Директория для сохранения.
        logging_steps: Частота логов.
        eval_strategy: Стратегия валидации (steps/epoch/no).
        eval_steps: Частота валидации.
        save_strategy: Стратегия сохранения.
        save_steps: Частота сохранения.
        save_total_limit: Максимум чекпоинтов.
        load_best_model_at_end: Загружать лучшую модель.
        metric_for_best_model: Метрика для выбора лучшей.
        greater_is_better: Больше = лучше.

        resume_from_checkpoint: Продолжить с чекпоинта.

        use_wandb: Использовать wandb.
        wandb_project: Проект wandb.
        wandb_mode: Режим wandb.
    """

    # Stage 1 checkpoint
    stage1_checkpoint: str = "models/qwen3_fashion_vocab/final"

    # Model settings
    max_seq_length: int = 2048
    dtype: Optional[torch.dtype] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    random_state: int = 1368
    num_proc: int = 8
    enable_thinking: bool = False

    # Data settings
    category: str = "Amazon_Fashion"
    data_dir: Path = field(default_factory=lambda: Path("data"))
    use_full_dataset: bool = True
    max_training_samples: Optional[int] = None
    eval_samples: int = 5000

    # Training params
    learning_rate: float = 1e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    gradient_clip_norm: float = 1.0
    max_steps: int = -1
    num_train_epochs: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    # Memory / optimizer
    gradient_checkpointing: bool = False
    optim: str = "adamw_8bit"

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("models/qwen3_fashion_full_finetuned"))
    logging_steps: int = 100
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    save_strategy: str = "steps"
    save_steps: int = 5000
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Resume
    resume_from_checkpoint: bool = False

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "semantic-id-full-finetune"
    wandb_mode: str = "online"

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
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        if not self.train_path.exists():
            raise FileNotFoundError(
                f"Training data not found: {self.train_path}. "
                f"Run data preparation for category '{self.category}'."
            )

        stage1_path = Path(self.stage1_checkpoint)
        if not stage1_path.exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found: {self.stage1_checkpoint}. "
                "Run Stage 1 vocabulary expansion first."
            )

        if not self.val_path.exists():
            logger.warning(f"Validation data not found: {self.val_path}")
            self.eval_strategy = "no"
            self.load_best_model_at_end = False

    def log_config(self) -> None:
        """Вывести конфигурацию в лог."""
        logger.info("=== Full Fine-tuning Configuration ===")

        logger.info("Stage 1 Checkpoint:")
        logger.info(f"  stage1_checkpoint: {self.stage1_checkpoint}")

        logger.info("Model Settings:")
        logger.info(f"  max_seq_length: {self.max_seq_length}")
        logger.info(f"  dtype: {self.dtype}")
        logger.info(f"  load_in_4bit: {self.load_in_4bit}")
        logger.info(f"  load_in_8bit: {self.load_in_8bit}")
        logger.info(f"  gradient_checkpointing: {self.gradient_checkpointing}")

        logger.info("Data Settings:")
        logger.info(f"  category: {self.category}")
        logger.info(f"  train_path: {self.train_path}")
        logger.info(f"  val_path: {self.val_path}")
        logger.info(f"  use_full_dataset: {self.use_full_dataset}")
        logger.info(f"  max_training_samples: {self.max_training_samples or 'All'}")
        logger.info(f"  eval_samples: {self.eval_samples}")

        logger.info("Training Parameters:")
        logger.info(f"  learning_rate: {self.learning_rate}")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        eff_batch = self.batch_size * self.gradient_accumulation_steps
        logger.info(f"  effective_batch_size: {eff_batch}")
        logger.info(f"  num_train_epochs: {self.num_train_epochs}")
        logger.info(f"  max_steps: {self.max_steps}")
        logger.info(f"  warmup_ratio: {self.warmup_ratio}")
        logger.info(f"  weight_decay: {self.weight_decay}")
        logger.info(f"  lr_scheduler_type: {self.lr_scheduler_type}")
        logger.info(f"  optim: {self.optim}")

        logger.info("Output Settings:")
        logger.info(f"  output_dir: {self.output_dir}")
        logger.info(f"  logging_steps: {self.logging_steps}")
        logger.info(f"  eval_strategy: {self.eval_strategy}")
        logger.info(f"  eval_steps: {self.eval_steps}")
        logger.info(f"  save_strategy: {self.save_strategy}")
        logger.info(f"  save_steps: {self.save_steps}")
        logger.info(f"  save_total_limit: {self.save_total_limit}")
        logger.info(f"  load_best_model_at_end: {self.load_best_model_at_end}")
        logger.info(f"  resume_from_checkpoint: {self.resume_from_checkpoint}")

        logger.info("Logging:")
        logger.info(f"  use_wandb: {self.use_wandb}")
        logger.info(f"  wandb_project: {self.wandb_project}")
        logger.info(f"  wandb_mode: {self.wandb_mode}")
        logger.info("======================================")

