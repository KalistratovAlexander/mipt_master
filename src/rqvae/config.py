from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("train-rqvae")


@dataclass
class RQVAEConfig:
    """Конфигурация обучения RQ-VAE."""

    # Data settings
    category: str = "Amazon_Fashion"
    data_dir: Path = field(default_factory=lambda: Path("amazon/data"))
    embeddings_path: Optional[Path] = None
    checkpoint_dir: Path = field(default_factory=lambda: Path("amazon/checkpoints") / "rqvae")

    # Model parameters
    item_embedding_dim: int = 1024
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    codebook_embedding_dim: int = 32
    codebook_quantization_levels: int = 3
    codebook_size: int = 256
    commitment_weight: float = 0.25
    use_rotation_trick: bool = True

    # EMA VQ
    use_ema_vq: bool = False
    ema_decay: float = 0.99
    ema_epsilon: float = 1e-5

    # Training parameters
    batch_size: int = 4096
    gradient_accumulation_steps: int = 1
    num_epochs: int = 20000
    scheduler_type: str = "cosine_with_warmup"  # "cosine" | "cosine_with_warmup" | "none"
    warmup_start_lr: float = 1e-8
    warmup_steps: int = 200
    max_lr: float = 3e-4
    min_lr: float = 1e-6
    use_gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0
    use_kmeans_init: bool = True
    reset_unused_codes: bool = True
    steps_per_codebook_reset: int = 2
    codebook_usage_threshold: float = 1.0
    val_split: float = 0.05

    # Logging and checkpointing
    steps_per_train_log: int = 10
    steps_per_val_log: int = 200

    # Reproducibility / runtime
    seed: int = 42

    def __post_init__(self):
        if self.embeddings_path is None:
            self.embeddings_path = self.data_dir / "output" / f"{self.category}_items_with_embeddings.parquet"

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if self.codebook_quantization_levels <= 0:
            raise ValueError("codebook_quantization_levels must be > 0")
        if self.codebook_size <= 0:
            raise ValueError("codebook_size must be > 0")
        if self.codebook_embedding_dim <= 0:
            raise ValueError("codebook_embedding_dim must be > 0")
        if not (0.0 <= self.val_split < 1.0):
            raise ValueError("val_split must be in [0, 1)")
        if self.scheduler_type not in {"cosine", "cosine_with_warmup", "none"}:
            raise ValueError("scheduler_type must be one of: cosine, cosine_with_warmup, none")

    def log_config(self) -> None:
        logger.info("=== RQ-VAE Configuration ===")
        logger.info("Data Settings:")
        logger.info(f"  category: {self.category}")
        logger.info(f"  data_dir: {self.data_dir}")
        logger.info(f"  embeddings_path: {self.embeddings_path}")
        logger.info(f"  checkpoint_dir: {self.checkpoint_dir}")

        logger.info("Model Parameters:")
        logger.info(f"  item_embedding_dim: {self.item_embedding_dim}")
        logger.info(f"  encoder_hidden_dims: {self.encoder_hidden_dims}")
        logger.info(f"  codebook_embedding_dim: {self.codebook_embedding_dim}")
        logger.info(f"  codebook_quantization_levels: {self.codebook_quantization_levels}")
        logger.info(f"  codebook_size: {self.codebook_size}")
        logger.info(f"  commitment_weight: {self.commitment_weight}")
        logger.info(f"  use_rotation_trick: {self.use_rotation_trick}")

        logger.info("EMA Settings:")
        logger.info(f"  use_ema_vq: {self.use_ema_vq}")
        logger.info(f"  ema_decay: {self.ema_decay}")
        logger.info(f"  ema_epsilon: {self.ema_epsilon}")

        logger.info("Training Parameters:")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        logger.info(f"  effective_batch_size: {self.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"  num_epochs: {self.num_epochs}")
        logger.info(f"  scheduler_type: {self.scheduler_type}")
        logger.info(f"  warmup_start_lr: {self.warmup_start_lr}")
        logger.info(f"  warmup_steps: {self.warmup_steps}")
        logger.info(f"  max_lr: {self.max_lr}")
        logger.info(f"  min_lr: {self.min_lr}")
        logger.info(f"  use_gradient_clipping: {self.use_gradient_clipping}")
        logger.info(f"  gradient_clip_norm: {self.gradient_clip_norm}")
        logger.info(f"  use_kmeans_init: {self.use_kmeans_init}")
        logger.info(f"  reset_unused_codes: {self.reset_unused_codes}")
        logger.info(f"  steps_per_codebook_reset: {self.steps_per_codebook_reset}")
        logger.info(f"  codebook_usage_threshold: {self.codebook_usage_threshold}")
        logger.info(f"  val_split: {self.val_split}")
        logger.info(f"  seed: {self.seed}")

        logger.info("Logging and Checkpointing:")
        logger.info(f"  steps_per_train_log: {self.steps_per_train_log}")
        logger.info(f"  steps_per_val_log: {self.steps_per_val_log}")
        logger.info("===========================")


