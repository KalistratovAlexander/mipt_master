"""Пакет для генерации эмбеддингов товаров.

Поддерживает несколько моделей:
- Qwen3-Embedding (default)
- jina-embeddings-v3
- bge-m3
- gte-Qwen2
- stella-en
- e5-large
- и другие

Использование CLI:
    # С preset
    python -m mipt_master.src.embed.cli --model-preset qwen3

    # С custom model
    python -m mipt_master.src.embed.cli --model-name "jinaai/jina-embeddings-v3" --pooling mean

    # Список моделей
    python -m mipt_master.src.embed.cli --list-models

Программный API:
    >>> from mipt_master.src.embed import EmbedConfig, embed_items, save_embeddings
    >>> from mipt_master.src.device import get_device_info
    >>> config = EmbedConfig(model_preset="jina-v3")
    >>> device_info = get_device_info()
    >>> result = embed_items(config, device_info)
    >>> save_embeddings(config, result, device_info)
"""

from .config import (
    EmbedConfig,
    ModelPreset,
    PoolingStrategy,
    MODEL_PRESETS,
    get_preset,
    list_presets,
)
from .dataset import TokenizedDataset, build_dataloader, load_tokenized_data
from .embed import EmbedResult, embed_items, save_embeddings
from .model import (
    cls_pool,
    generate_embeddings,
    get_pooling_fn,
    last_token_pool,
    load_model,
    mean_pool,
)

__all__ = [
    # Config
    "EmbedConfig",
    "ModelPreset",
    "PoolingStrategy",
    "MODEL_PRESETS",
    "get_preset",
    "list_presets",
    # Dataset
    "TokenizedDataset",
    "load_tokenized_data",
    "build_dataloader",
    # Model
    "load_model",
    "last_token_pool",
    "mean_pool",
    "cls_pool",
    "get_pooling_fn",
    "generate_embeddings",
    # Embed
    "EmbedResult",
    "embed_items",
    "save_embeddings",
]
