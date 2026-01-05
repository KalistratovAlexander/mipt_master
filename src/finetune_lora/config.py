"""Конфигурация для Stage 2: LoRA fine-tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("finetune-lora")


def is_bf16_supported() -> bool:
    """Проверить поддержку bfloat16."""
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    return False


# Стандартные target modules для Qwen/LLaMA
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class LoRAConfig:
    """Конфигурация LoRA fine-tuning.

    Stage 2: Обучаем LoRA-адаптеры на модели с расширенным словарём из Stage 1.

    Attributes:
        stage1_checkpoint: Путь к чекпоинту Stage 1 (с расширенным словарём).
        category: Категория товаров.
        data_dir: Директория с данными.
        max_seq_length: Максимальная длина последовательности.
        num_proc: Число процессов для обработки данных.
        enable_thinking: Включить thinking mode (Qwen3).

        use_full_dataset: Использовать полный датасет.
        max_training_samples: Максимум примеров для обучения (если не full).
        eval_samples: Число примеров для валидации.

        lora_r: Rank LoRA.
        lora_alpha: Alpha LoRA.
        lora_dropout: Dropout LoRA.
        lora_target_modules: Список модулей для LoRA (None = дефолт).
        use_rslora: Использовать RSLoRA.

        batch_size: Размер батча.
        gradient_accumulation_steps: Шаги аккумуляции.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        lr_scheduler_type: Тип scheduler.
        warmup_ratio: Доля warmup от total steps.
        gradient_clip_norm: Gradient clipping.
        num_train_epochs: Число эпох.
        max_steps: Максимум шагов (-1 = по эпохам).

        load_in_4bit: Загружать модель в 4-bit (QLoRA).
        dtype: Тип данных для вычислений.
        gradient_checkpointing: Gradient checkpointing.

        output_dir: Директория для сохранения.
        logging_steps: Частота логов.
        eval_steps: Частота валидации.
        save_steps: Частота сохранения.
        save_total_limit: Максимум сохранённых чекпоинтов.
        random_state: Seed.

        use_wandb: Использовать wandb.
        wandb_project: Проект wandb.
        wandb_mode: Режим wandb.

        save_merged: Сохранять merged модель.
    """

    # Stage 1 checkpoint
    stage1_checkpoint: str = "models/qwen3_fashion_vocab/final"

    # Data settings
    category: str = "Amazon_Fashion"
    data_dir: Path = field(default_factory=lambda: Path("data"))
    max_seq_length: int = 1024
    num_proc: int = 8
    enable_thinking: bool = False

    # Training data sampling
    use_full_dataset: bool = True
    max_training_samples: Optional[int] = None
    eval_samples: int = 5000

    # LoRA params
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list[str]] = None
    use_rslora: bool = True

    # Optimization
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    gradient_clip_norm: float = 1.0
    num_train_epochs: int = 2
    max_steps: int = -1

    # Precision / memory
    load_in_4bit: bool = True
    dtype: Optional[torch.dtype] = None
    gradient_checkpointing: bool = False

    # Logging / saving
    output_dir: Path = field(default_factory=lambda: Path("models/qwen3_fashion_lora"))
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    random_state: int = 1368

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "semantic-id-lora-finetune"
    wandb_mode: str = "online"

    # Save options
    save_merged: bool = True

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

        if self.lora_target_modules is None:
            self.lora_target_modules = DEFAULT_LORA_TARGET_MODULES.copy()

    def validate(self) -> None:
        """Проверить валидность конфигурации."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        if self.lora_r <= 0:
            raise ValueError("lora_r must be > 0")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be > 0")

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

    def log_config(self) -> None:
        """Вывести конфигурацию в лог."""
        logger.info("=== Stage 2 LoRA Configuration ===")

        logger.info("Stage 1 Checkpoint:")
        logger.info(f"  stage1_checkpoint: {self.stage1_checkpoint}")

        logger.info("Data Settings:")
        logger.info(f"  category: {self.category}")
        logger.info(f"  train_path: {self.train_path}")
        logger.info(f"  val_path: {self.val_path}")
        logger.info(f"  max_seq_length: {self.max_seq_length}")
        logger.info(f"  use_full_dataset: {self.use_full_dataset}")
        logger.info(f"  max_training_samples: {self.max_training_samples}")
        logger.info(f"  eval_samples: {self.eval_samples}")

        logger.info("LoRA Parameters:")
        logger.info(f"  lora_r: {self.lora_r}")
        logger.info(f"  lora_alpha: {self.lora_alpha}")
        logger.info(f"  lora_dropout: {self.lora_dropout}")
        logger.info(f"  lora_target_modules: {self.lora_target_modules}")
        logger.info(f"  use_rslora: {self.use_rslora}")

        logger.info("Training Parameters:")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        eff_batch = self.batch_size * self.gradient_accumulation_steps
        logger.info(f"  effective_batch_size: {eff_batch}")
        logger.info(f"  learning_rate: {self.learning_rate}")
        logger.info(f"  weight_decay: {self.weight_decay}")
        logger.info(f"  lr_scheduler_type: {self.lr_scheduler_type}")
        logger.info(f"  warmup_ratio: {self.warmup_ratio}")
        logger.info(f"  num_train_epochs: {self.num_train_epochs}")
        logger.info(f"  max_steps: {self.max_steps}")

        logger.info("Memory Settings:")
        logger.info(f"  load_in_4bit: {self.load_in_4bit}")
        logger.info(f"  dtype: {self.dtype}")
        logger.info(f"  gradient_checkpointing: {self.gradient_checkpointing}")

        logger.info("Output Settings:")
        logger.info(f"  output_dir: {self.output_dir}")
        logger.info(f"  logging_steps: {self.logging_steps}")
        logger.info(f"  eval_steps: {self.eval_steps}")
        logger.info(f"  save_steps: {self.save_steps}")
        logger.info(f"  save_merged: {self.save_merged}")

        logger.info("Logging:")
        logger.info(f"  use_wandb: {self.use_wandb}")
        logger.info(f"  wandb_project: {self.wandb_project}")
        logger.info(f"  wandb_mode: {self.wandb_mode}")
        logger.info("==================================")

