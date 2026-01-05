"""CLI для Stage 2: LoRA fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from mipt_master.src.device import get_device_info
from mipt_master.src.logger import setup_logger

logger = setup_logger("finetune-lora", log_to_file=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Парсинг аргументов командной строки.

    Args:
        argv: Аргументы (None = sys.argv).

    Returns:
        Распарсенные аргументы.
    """
    p = argparse.ArgumentParser(
        description="Stage 2: LoRA fine-tuning on semantic ID conversations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage 1 checkpoint
    p.add_argument(
        "--stage1-checkpoint", type=str, default="models/qwen3_fashion_vocab/final",
        help="Path to Stage 1 checkpoint with extended vocabulary",
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
        "--max-seq-length", type=int, default=1024,
        help="Maximum sequence length",
    )
    data_group.add_argument(
        "--use-full-dataset", action="store_true", default=True,
        help="Use full training dataset",
    )
    data_group.add_argument(
        "--max-training-samples", type=int, default=None,
        help="Max training samples (if not full)",
    )
    data_group.add_argument(
        "--eval-samples", type=int, default=5000,
        help="Number of validation samples",
    )
    data_group.add_argument(
        "--num-proc", type=int, default=8,
        help="Number of processes for data loading",
    )

    # LoRA settings
    lora_group = p.add_argument_group("LoRA")
    lora_group.add_argument(
        "--lora-r", type=int, default=16,
        help="LoRA rank",
    )
    lora_group.add_argument(
        "--lora-alpha", type=int, default=16,
        help="LoRA alpha",
    )
    lora_group.add_argument(
        "--lora-dropout", type=float, default=0.05,
        help="LoRA dropout",
    )
    lora_group.add_argument(
        "--no-rslora", action="store_true",
        help="Disable RSLoRA",
    )

    # Training settings
    train_group = p.add_argument_group("Training")
    train_group.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-device batch size",
    )
    train_group.add_argument(
        "--gradient-accumulation-steps", type=int, default=4,
        help="Gradient accumulation steps",
    )
    train_group.add_argument(
        "--lr", "--learning-rate", type=float, default=2e-4, dest="learning_rate",
        help="Learning rate",
    )
    train_group.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="Weight decay",
    )
    train_group.add_argument(
        "--lr-scheduler", type=str, default="cosine", dest="lr_scheduler_type",
        help="Learning rate scheduler",
    )
    train_group.add_argument(
        "--warmup-ratio", type=float, default=0.03,
        help="Warmup ratio",
    )
    train_group.add_argument(
        "--gradient-clip-norm", type=float, default=1.0,
        help="Gradient clipping norm",
    )
    train_group.add_argument(
        "--epochs", type=int, default=2, dest="num_train_epochs",
        help="Number of training epochs",
    )
    train_group.add_argument(
        "--max-steps", type=int, default=-1,
        help="Max training steps (-1 = use epochs)",
    )
    train_group.add_argument(
        "--seed", type=int, default=1368, dest="random_state",
        help="Random seed",
    )

    # Memory settings
    mem_group = p.add_argument_group("Memory")
    mem_group.add_argument(
        "--load-in-4bit", action="store_true", default=True,
        help="Load model in 4-bit (QLoRA)",
    )
    mem_group.add_argument(
        "--no-4bit", action="store_true",
        help="Disable 4-bit loading",
    )
    mem_group.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Enable gradient checkpointing",
    )

    # Output settings
    output_group = p.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir", type=Path, default=Path("models/qwen3_fashion_lora"),
        help="Output directory",
    )
    output_group.add_argument(
        "--logging-steps", type=int, default=50,
        help="Log every N steps",
    )
    output_group.add_argument(
        "--eval-steps", type=int, default=500,
        help="Evaluate every N steps",
    )
    output_group.add_argument(
        "--save-steps", type=int, default=1000,
        help="Save checkpoint every N steps",
    )
    output_group.add_argument(
        "--save-total-limit", type=int, default=3,
        help="Max checkpoints to keep",
    )
    output_group.add_argument(
        "--no-save-merged", action="store_true",
        help="Don't save merged model",
    )

    # Logging settings
    log_group = p.add_argument_group("Logging")
    log_group.add_argument(
        "--no-wandb", action="store_true",
        help="Disable wandb logging",
    )
    log_group.add_argument(
        "--wandb-project", type=str, default="semantic-id-lora-finetune",
        help="Wandb project name",
    )
    log_group.add_argument(
        "--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"],
        help="Wandb mode",
    )

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Точка входа для LoRA fine-tuning."""
    args = parse_args(argv)

    from .config import LoRAConfig
    from .model import load_model_with_lora
    from .train import (
        finish_wandb,
        load_conversation_dataset,
        save_lora_adapter,
        save_merged_model,
        train_lora,
    )

    # Пытаемся импортировать тестовые промпты
    try:
        from mipt_master.src.test_prompts import REC_TEST_PROMPTS, TEST_PROMPTS
    except ImportError:
        logger.warning("Could not import test prompts, using empty lists")
        TEST_PROMPTS = []
        REC_TEST_PROMPTS = []

    # Создаём конфигурацию
    config = LoRAConfig(
        stage1_checkpoint=args.stage1_checkpoint,
        category=args.category,
        data_dir=args.data_dir,
        max_seq_length=args.max_seq_length,
        num_proc=args.num_proc,
        use_full_dataset=args.use_full_dataset,
        max_training_samples=args.max_training_samples,
        eval_samples=args.eval_samples,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=not args.no_rslora,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        gradient_clip_norm=args.gradient_clip_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        load_in_4bit=not args.no_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        random_state=args.random_state,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        save_merged=not args.no_save_merged,
    )

    # Валидация
    config.validate()

    # Device info
    device_info = get_device_info()
    logger.info(f"Device: {device_info.device}")

    # Логируем конфиг
    config.log_config()

    # Обучаем
    result = train_lora(
        config=config,
        generic_prompts=TEST_PROMPTS,
        sid_prompts=REC_TEST_PROMPTS,
    )

    # Загружаем модель для сохранения
    # (trainer уже сохранил чекпоинты, но нам нужен финальный adapter)
    model, tokenizer = load_model_with_lora(config)

    # Сохраняем LoRA adapter
    adapter_path = save_lora_adapter(model, tokenizer, config)
    logger.info(f"LoRA adapter saved to: {adapter_path}")

    # Сохраняем merged модель если включено
    if config.save_merged:
        merged_path = save_merged_model(model, tokenizer, config)
        if merged_path:
            logger.info(f"Merged model saved to: {merged_path}")

    # Завершаем wandb
    finish_wandb()

    logger.info("=" * 50)
    logger.info("Stage 2: LoRA fine-tuning complete!")
    logger.info(f"Training loss: {result.training_loss}")
    logger.info(f"Total steps: {result.global_step}")
    logger.info(f"Runtime: {result.runtime_seconds:.1f}s")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

