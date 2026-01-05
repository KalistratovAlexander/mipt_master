"""CLI для генерации эмбеддингов."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

# Disable tokenizers parallelism to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mipt_master.src.device import get_device_info
from mipt_master.src.logger import setup_logger

from .config import EmbedConfig, PoolingStrategy, list_presets, MODEL_PRESETS
from .embed import embed_items, save_embeddings

logger = setup_logger("embed-items", log_to_file=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    epilog = list_presets()

    p = argparse.ArgumentParser(
        description="Generate embeddings for items using various embedding models",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data settings
    p.add_argument("--category", type=str, default="Amazon_Fashion",
                   help="Product category")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Data directory")
    p.add_argument("--input-path", type=Path, default=None,
                   help="Input parquet path (default: auto)")
    p.add_argument("--output-path", type=Path, default=None,
                   help="Output parquet path (default: auto)")
    p.add_argument("--tokenized-path", type=Path, default=None,
                   help="Pre-tokenized .npz path (default: auto)")
    p.add_argument("--num-rows", type=int, default=None,
                   help="Limit number of rows (default: all)")

    # Model settings
    model_group = p.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model-preset", type=str, default="qwen3",
        choices=list(MODEL_PRESETS.keys()),
        help="Model preset (default: qwen3). See list below.",
    )
    model_group.add_argument(
        "--model-name", type=str, default=None,
        help="Custom HuggingFace model name (overrides preset)",
    )

    p.add_argument("--pooling", type=str, default=None,
                   choices=["last_token", "mean", "cls"],
                   help="Pooling strategy (default: from preset)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for inference")
    p.add_argument("--target-dim", type=int, default=None,
                   help="Target embedding dimension (default: from preset)")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Allow trust_remote_code for model loading")

    # Runtime settings
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader num_workers")
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile")
    p.add_argument("--verify-consistency", action="store_true",
                   help="Verify single vs batch embedding consistency")
    p.add_argument("--log-freq", type=int, default=1000,
                   help="Log progress every N items")

    # Utility
    p.add_argument("--list-models", action="store_true",
                   help="List available model presets and exit")

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Точка входа для генерации эмбеддингов."""
    args = parse_args(argv)

    # Показать список моделей и выйти
    if args.list_models:
        print(list_presets())
        return

    # Создаём конфигурацию
    config_kwargs = {
        "category": args.category,
        "data_dir": args.data_dir,
        "input_path": args.input_path,
        "output_path": args.output_path,
        "tokenized_path": args.tokenized_path,
        "num_rows": args.num_rows,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "use_compile": not args.no_compile,
        "verify_consistency": args.verify_consistency,
        "log_freq": args.log_freq,
    }

    # Model settings: preset или custom
    if args.model_name is not None:
        # Custom model — нужно явно указать pooling
        config_kwargs["model_name"] = args.model_name
        if args.pooling is None:
            logger.warning("Using custom model without --pooling. Defaulting to last_token.")
            config_kwargs["pooling_strategy"] = PoolingStrategy.LAST_TOKEN
        else:
            config_kwargs["pooling_strategy"] = PoolingStrategy(args.pooling)
        if args.target_dim is not None:
            config_kwargs["target_dim"] = args.target_dim
        if args.trust_remote_code:
            config_kwargs["trust_remote_code"] = True
    else:
        # Используем preset
        config_kwargs["model_preset"] = args.model_preset

        # Переопределения из CLI
        if args.pooling is not None:
            config_kwargs["pooling_strategy"] = PoolingStrategy(args.pooling)
        if args.target_dim is not None:
            config_kwargs["target_dim"] = args.target_dim

    config = EmbedConfig(**config_kwargs)

    # Валидация
    config.validate()

    # Device info
    device_info = get_device_info()
    logger.info(f"Device: {device_info.device}")

    # Выводим конфигурацию
    config.log_config()

    # Генерируем эмбеддинги
    result = embed_items(config, device_info)

    # Сохраняем результат
    save_embeddings(config, result, device_info)

    logger.info("Done!")


if __name__ == "__main__":
    main()
