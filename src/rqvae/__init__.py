"""RQ-VAE (Residual Quantized VAE) для генерации семантических ID.

Этот пакет содержит компоненты RQ-VAE:
- config: конфигурация модели и обучения
- data: загрузка данных, split, DataLoader'ы
- model: модель RQVAE
- metrics: метрики качества (unique IDs, codebook usage, residual norm)
- train: тренировочный цикл
- cli: интерфейс командной строки
"""

from .config import RQVAEConfig
from .data import build_loaders, load_embeddings, prepare_data
from .metrics import avg_residual_norm, codebook_usage, unique_ids_proportion
from .model import ForwardOutput, RQVAE
from .quantization import EMAVectorQuantizer, QuantizationOutput, VectorQuantizer
from .train import TrainState, train_rqvae

__all__ = [
    # Config
    "RQVAEConfig",
    # Data
    "load_embeddings",
    "build_loaders",
    "prepare_data",
    # Model
    "RQVAE",
    "ForwardOutput",
    # Quantization
    "QuantizationOutput",
    "VectorQuantizer",
    "EMAVectorQuantizer",
    # Metrics
    "unique_ids_proportion",
    "codebook_usage",
    "avg_residual_norm",
    # Train
    "TrainState",
    "train_rqvae",
]
