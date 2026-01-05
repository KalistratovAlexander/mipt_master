"""CLI для vocabulary expansion (Stage 1)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from mipt_master.src.device import get_device_info
from mipt_master.src.logger import setup_logger

logger = setup_logger("vocab-expansion", log_to_file=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Парсинг аргументов командной строки.

    Args:
        argv: Аргументы (None = sys.argv).

    Returns:
        Распарсенные аргументы.
    """
    p = argparse.ArgumentParser(
        description="Stage 1: Vocabulary expansion and embedding initialization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model settings
    model_group = p.add_argument_group("Model")
    model_group.add_argument(
        "--model-name", type=str, default="unsloth/Qwen3-1.7B",
        help="Base model name (HuggingFace or unsloth)",
    )
    model_group.add_argument(
        "--max-seq-length", type=int, default=1024,
        help="Maximum sequence length",
    )
    model_group.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load model in 4-bit quantization",
    )
    model_group.add_argument(
        "--load-in-8bit", action="store_true",
        help="Load model in 8-bit quantization",
    )
    model_group.add_argument(
        "--no-gradient-checkpointing", action="store_true",
        help="Disable gradient checkpointing",
    )

    # Vocabulary settings
    vocab_group = p.add_argument_group("Vocabulary")
    vocab_group.add_argument(
        "--no-extend-vocab", action="store_true",
        help="Skip vocabulary extension",
    )
    vocab_group.add_argument(
        "--codebook-levels", type=int, default=4,
        help="Number of hierarchical levels",
    )
    vocab_group.add_argument(
        "--codebook-size", type=int, default=256,
        help="Codebook size per level",
    )
    vocab_group.add_argument(
        "--num-semantic-tokens", type=int, default=1024,
        help="Number of semantic ID tokens",
    )

    # Data settings
    data_group = p.add_argument_group("Data")
    data_group.add_argument(
        "--category", type=str, default="Amazon_Fashion",
        help="Product category",
    )
    data_group.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Data directory",
    )
    data_group.add_argument(
        "--max-training-samples", type=int, default=16000,
        help="Maximum training samples",
    )
    data_group.add_argument(
        "--val-samples", type=int, default=500,
        help="Number of validation samples",
    )
    data_group.add_argument(
        "--num-proc", type=int, default=8,
        help="Number of processes for data loading",
    )

    # Training settings
    train_group = p.add_argument_group("Training")
    train_group.add_argument(
        "--lr", "--learning-rate", type=float, default=1e-3, dest="learning_rate",
        help="Learning rate",
    )
    train_group.add_argument(
        "--batch-size", type=int, default=1,
        help="Per-device batch size",
    )
    train_group.add_argument(
        "--gradient-accumulation-steps", type=int, default=4,
        help="Gradient accumulation steps",
    )
    train_group.add_argument(
        "--max-steps", type=int, default=1000,
        help="Maximum training steps (0 = use epochs)",
    )
    train_group.add_argument(
        "--epochs", type=int, default=1, dest="num_train_epochs",
        help="Number of training epochs",
    )
    train_group.add_argument(
        "--warmup-steps", type=int, default=100,
        help="Warmup steps",
    )
    train_group.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="Weight decay",
    )
    train_group.add_argument(
        "--lr-scheduler", type=str, default="cosine", dest="lr_scheduler_type",
        help="Learning rate scheduler type",
    )
    train_group.add_argument(
        "--optim", type=str, default="adamw_8bit",
        help="Optimizer type",
    )
    train_group.add_argument(
        "--seed", type=int, default=1368, dest="random_state",
        help="Random seed",
    )

    # Output settings
    output_group = p.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir", type=Path, default=Path("models/qwen3_fashion_vocab"),
        help="Output directory",
    )
    output_group.add_argument(
        "--steps-per-train-log", type=int, default=100,
        help="Log training every N steps",
    )
    output_group.add_argument(
        "--steps-per-val-log", type=int, default=250,
        help="Evaluate every N steps",
    )
    output_group.add_argument(
        "--save-steps", type=int, default=5000,
        help="Save checkpoint every N steps",
    )

    # Logging settings
    log_group = p.add_argument_group("Logging")
    log_group.add_argument(
        "--no-wandb", action="store_true",
        help="Disable wandb logging",
    )
    log_group.add_argument(
        "--wandb-project", type=str, default="semantic-id-vocab-extension",
        help="Wandb project name",
    )
    log_group.add_argument(
        "--wandb-mode", type=str, default="offline", choices=["online", "offline", "disabled"],
        help="Wandb mode",
    )

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Точка входа для vocabulary expansion."""
    args = parse_args(argv)

    # Unsloth должен быть импортирован первым
    from unsloth import FastLanguageModel

    from .config import VocabExpansionConfig
    from .tokenizer import extend_tokenizer, prepare_model_for_embedding_training
    from .train import save_embeddings_artifact, save_model_and_tokenizer, train_embeddings

    # Пытаемся импортировать тестовые промпты
    try:
        from mipt_master.src.test_prompts import REC_TEST_PROMPTS
    except ImportError:
        logger.warning("Could not import REC_TEST_PROMPTS, using empty list")
        REC_TEST_PROMPTS = []

    # Создаём конфигурацию
    config = VocabExpansionConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        random_state=args.random_state,
        num_proc=args.num_proc,
        extend_vocabulary=not args.no_extend_vocab,
        codebook_levels=args.codebook_levels,
        codebook_size=args.codebook_size,
        num_semantic_tokens=args.num_semantic_tokens,
        category=args.category,
        data_dir=args.data_dir,
        max_training_samples=args.max_training_samples,
        val_samples=args.val_samples,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        optim=args.optim,
        output_dir=args.output_dir,
        steps_per_train_log=args.steps_per_train_log,
        steps_per_val_log=args.steps_per_val_log,
        save_steps=args.save_steps,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
    )

    # Валидация
    config.validate()

    # Device info
    device_info = get_device_info()
    logger.info(f"Device: {device_info.device}")

    # Логируем конфиг
    config.log_config()

    # Инициализируем wandb
    if config.use_wandb:
        try:
            import wandb

            run_name = f"vocab-ext-{config.category}-lr{config.learning_rate}"
            wandb.init(
                project=config.wandb_project,
                name=run_name,
                config={
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in config.__dict__.items()
                    if not k.startswith("_")
                },
                mode=config.wandb_mode,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            config.use_wandb = False

    # Загружаем модель
    logger.info(f"Loading base model: {config.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )

    # Расширяем словарь
    num_new_tokens = 0
    if config.extend_vocabulary:
        num_new_tokens = extend_tokenizer(model, tokenizer, config)
        model = prepare_model_for_embedding_training(model, tokenizer, config, num_new_tokens)

    # Обучаем
    train_result = train_embeddings(
        model=model,
        tokenizer=tokenizer,
        config=config,
        num_new_tokens=num_new_tokens,
        test_prompts=REC_TEST_PROMPTS,
    )

    # Сохраняем эмбеддинги как артефакт
    if num_new_tokens > 0:
        save_embeddings_artifact(model, tokenizer, config, num_new_tokens, train_result)

    # Сохраняем модель
    save_model_and_tokenizer(model, tokenizer, config, train_result)

    # Завершаем wandb
    if config.use_wandb:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass

    logger.info("=" * 50)
    logger.info("Stage 1: Vocabulary expansion complete!")
    logger.info(f"Initialized {config.num_semantic_tokens + 3} new tokens")
    logger.info(f"Model saved to: {config.output_dir / 'final'}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

