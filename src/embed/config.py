"""Конфигурация для генерации эмбеддингов."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger("embed-items")


class PoolingStrategy(str, Enum):
    """Стратегия pooling для извлечения эмбеддингов."""

    LAST_TOKEN = "last_token"  # Qwen, GTE, некоторые другие
    MEAN = "mean"  # Jina, Stella, E5
    CLS = "cls"  # BGE, BERT-based


@dataclass
class ModelPreset:
    """Пресет модели с настройками по умолчанию.

    Attributes:
        name: HuggingFace model name.
        pooling: Стратегия pooling.
        default_dim: Размерность эмбеддинга по умолчанию.
        max_dim: Максимальная размерность.
        trust_remote_code: Требуется ли trust_remote_code.
        description: Краткое описание.
    """

    name: str
    pooling: PoolingStrategy
    default_dim: int
    max_dim: int
    trust_remote_code: bool = False
    description: str = ""


# Поддерживаемые модели
MODEL_PRESETS: dict[str, ModelPreset] = {
    "qwen3": ModelPreset(
        name="Qwen/Qwen3-Embedding-0.6B",
        pooling=PoolingStrategy.LAST_TOKEN,
        default_dim=1024,
        max_dim=1024,
        description="Alibaba Qwen3, 600M params, multilingual, Matryoshka",
    ),
    "qwen3-large": ModelPreset(
        name="Qwen/Qwen3-Embedding-4B",
        pooling=PoolingStrategy.LAST_TOKEN,
        default_dim=2560,
        max_dim=2560,
        description="Alibaba Qwen3 Large, 4B params, better quality",
    ),
    "gte-qwen2": ModelPreset(
        name="Alibaba-NLP/gte-Qwen2-7B-instruct",
        pooling=PoolingStrategy.LAST_TOKEN,
        default_dim=3584,
        max_dim=3584,
        trust_remote_code=True,
        description="Alibaba GTE-Qwen2, 7B params, SOTA quality",
    ),
    "jina-v3": ModelPreset(
        name="jinaai/jina-embeddings-v3",
        pooling=PoolingStrategy.MEAN,
        default_dim=1024,
        max_dim=1024,
        trust_remote_code=True,
        description="Jina v3, 570M params, multilingual, Matryoshka",
    ),
    "bge-m3": ModelPreset(
        name="BAAI/bge-m3",
        pooling=PoolingStrategy.CLS,
        default_dim=1024,
        max_dim=1024,
        description="BGE-M3, 568M params, multilingual, dense+sparse",
    ),
    "bge-large-en": ModelPreset(
        name="BAAI/bge-large-en-v1.5",
        pooling=PoolingStrategy.CLS,
        default_dim=1024,
        max_dim=1024,
        description="BGE Large English, 335M params",
    ),
    "stella-en": ModelPreset(
        name="dunzhang/stella_en_400M_v5",
        pooling=PoolingStrategy.MEAN,
        default_dim=1024,
        max_dim=1024,
        trust_remote_code=True,
        description="Stella EN, 400M params, English only, fast",
    ),
    "e5-large": ModelPreset(
        name="intfloat/e5-large-v2",
        pooling=PoolingStrategy.MEAN,
        default_dim=1024,
        max_dim=1024,
        description="E5 Large v2, 335M params, good quality",
    ),
    "e5-mistral": ModelPreset(
        name="intfloat/e5-mistral-7b-instruct",
        pooling=PoolingStrategy.LAST_TOKEN,
        default_dim=4096,
        max_dim=4096,
        description="E5 Mistral, 7B params, instruction-tuned",
    ),
    "mxbai-large": ModelPreset(
        name="mixedbread-ai/mxbai-embed-large-v1",
        pooling=PoolingStrategy.CLS,
        default_dim=1024,
        max_dim=1024,
        description="MixedBread Large, 335M params, very fast",
    ),
    "nomic-v1.5": ModelPreset(
        name="nomic-ai/nomic-embed-text-v1.5",
        pooling=PoolingStrategy.MEAN,
        default_dim=768,
        max_dim=768,
        trust_remote_code=True,
        description="Nomic Embed, 137M params, Matryoshka",
    ),
}


def get_preset(name: str) -> ModelPreset:
    """Получить пресет модели по имени.

    Args:
        name: Имя пресета (например, "qwen3", "jina-v3").

    Returns:
        ModelPreset с настройками.

    Raises:
        ValueError: Если пресет не найден.
    """
    if name not in MODEL_PRESETS:
        available = ", ".join(MODEL_PRESETS.keys())
        raise ValueError(f"Unknown model preset: {name}. Available: {available}")
    return MODEL_PRESETS[name]


def list_presets() -> str:
    """Вернуть строку с описанием всех пресетов."""
    lines = ["Available model presets:"]
    for key, preset in MODEL_PRESETS.items():
        lines.append(f"  {key:15} - {preset.description}")
    return "\n".join(lines)


@dataclass
class EmbedConfig:
    """Конфигурация генерации эмбеддингов.

    Attributes:
        category: Категория товаров (например, "Amazon_Fashion").
        data_dir: Директория с данными.
        input_path: Путь к исходному parquet-файлу (автогенерируется если None).
        output_path: Путь для сохранения эмбеддингов (автогенерируется если None).
        tokenized_path: Путь к pre-tokenized данным (автогенерируется если None).
        num_rows: Ограничение числа строк (None = все).
        model_preset: Имя пресета модели (qwen3, jina-v3, bge-m3, ...).
        model_name: HuggingFace model name (переопределяет preset).
        pooling_strategy: Стратегия pooling (автоматически из preset).
        batch_size: Размер батча для инференса.
        target_dim: Целевая размерность эмбеддинга.
        trust_remote_code: Разрешить remote code (автоматически из preset).
        num_workers: Число воркеров DataLoader.
        prefetch_factor: Prefetch factor для DataLoader.
        use_compile: Использовать torch.compile (только CUDA).
        verify_consistency: Проверять consistency single vs batch.
        log_freq: Частота логирования (каждые N items).
    """

    # Data settings
    category: str = "Amazon_Fashion"
    data_dir: Path = field(default_factory=lambda: Path("data"))
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    tokenized_path: Optional[Path] = None
    num_rows: Optional[int] = None

    # Model settings
    model_preset: str = "qwen3"
    model_name: Optional[str] = None  # Переопределяет preset
    pooling_strategy: PoolingStrategy = PoolingStrategy.LAST_TOKEN
    batch_size: int = 64
    target_dim: int = 1024
    trust_remote_code: bool = False

    # DataLoader settings
    num_workers: int = 4
    prefetch_factor: int = 2

    # Runtime settings
    use_compile: bool = True
    verify_consistency: bool = False
    log_freq: int = 1000

    def __post_init__(self) -> None:
        """Автогенерация путей и применение preset."""
        self.data_dir = Path(self.data_dir)

        # Применяем preset если model_name не задан явно
        if self.model_name is None:
            preset = get_preset(self.model_preset)
            self.model_name = preset.name
            self.pooling_strategy = preset.pooling
            self.target_dim = preset.default_dim
            self.trust_remote_code = preset.trust_remote_code

        # Автогенерация путей
        if self.input_path is None:
            self.input_path = self.data_dir / "output" / f"{self.category}_items_updated.parquet"

        if self.output_path is None:
            self.output_path = self.data_dir / "output" / f"{self.category}_items_with_embeddings.parquet"

        if self.tokenized_path is None:
            suffix = f"_{self.num_rows}" if self.num_rows else ""
            self.tokenized_path = self.data_dir / "output" / f"{self.category}_tokenized{suffix}.npz"

    def validate(self) -> None:
        """Проверить валидность конфигурации."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.target_dim <= 0:
            raise ValueError("target_dim must be > 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        if not self.tokenized_path.exists():
            raise FileNotFoundError(
                f"Pre-tokenized data not found: {self.tokenized_path}. "
                "Please run tokenize_items.py first."
            )

    def log_config(self) -> None:
        """Вывести конфигурацию в лог."""
        logger.info("=== Embed Configuration ===")
        logger.info("Data Settings:")
        logger.info(f"  category: {self.category}")
        logger.info(f"  data_dir: {self.data_dir}")
        logger.info(f"  input_path: {self.input_path}")
        logger.info(f"  output_path: {self.output_path}")
        logger.info(f"  tokenized_path: {self.tokenized_path}")
        logger.info(f"  num_rows: {self.num_rows or 'all'}")

        logger.info("Model Settings:")
        logger.info(f"  model_preset: {self.model_preset}")
        logger.info(f"  model_name: {self.model_name}")
        logger.info(f"  pooling_strategy: {self.pooling_strategy.value}")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  target_dim: {self.target_dim}")
        logger.info(f"  trust_remote_code: {self.trust_remote_code}")

        logger.info("Runtime Settings:")
        logger.info(f"  num_workers: {self.num_workers}")
        logger.info(f"  prefetch_factor: {self.prefetch_factor}")
        logger.info(f"  use_compile: {self.use_compile}")
        logger.info(f"  verify_consistency: {self.verify_consistency}")
        logger.info(f"  log_freq: {self.log_freq}")
        logger.info("===========================")
