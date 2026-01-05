"""CLI для Stage 2: Full fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from mipt_master.src.device import get_device_info
from mipt_master.src.logger import setup_logger

logger = setup_logger("finetune-full", log_to_file=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Парсинг аргументов командной строки.

    Args:
        argv: Аргументы (None = sys.argv).

    Returns:
        Распарсенные аргументы.
    """
    p = argparse.ArgumentParser(
        description="Stage 2: Full fine-tuning on semantic ID conversations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage 1 checkpoint
    p.add_argument(
        "--stage1-checkpoint", type=str, default="models/qwen3_fashion_vocab/final",
        help="Path to Stage 1 checkpoint with extended vocabulary",
    )

    # Model settings
    model_group = p.add_argument_group("Model")
    model_group.add_argument(
        "--max-seq-length", type=int, default=2048,
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
        "--gradient-checkpointing", action="store_true",
        help="Enable gradient checkpointing",
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

    # Training settings
    train_group = p.add_argument_group("Training")
    train_group.add_argument(
        "--lr", "--learning-rate", type=float, default=1e-5, dest="learning_rate",
        help="Learning rate",
    )
    train_group.add_argument(
        "--batch-size", type=int, default=8,
        help="Per-device batch size",
    )
    train_group.add_argument(
        "--gradient-accumulation-steps", type=int, default=8,
        help="Gradient accumulation steps",
    )
    train_group.add_argument(
        "--gradient-clip-norm", type=float, default=1.0,
        help="Gradient clipping norm",
    )
    train_group.add_argument(
        "--epochs", type=int, default=1, dest="num_train_epochs",
        help="Number of training epochs",
    )
    train_group.add_argument(
        "--max-steps", type=int, default=-1,
        help="Max training steps (-1 = use epochs)",
    )
    train_group.add_argument(
        "--warmup-ratio", type=float, default=0.03,
        help="Warmup ratio",
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
        "--output-dir", type=Path, default=Path("models/qwen3_fashion_full_finetuned"),
        help="Output directory",
    )
    output_group.add_argument(
        "--logging-steps", type=int, default=100,
        help="Log every N steps",
    )
    output_group.add_argument(
        "--eval-strategy", type=str, default="steps", choices=["steps", "epoch", "no"],
        help="Evaluation strategy",
    )
    output_group.add_argument(
        "--eval-steps", type=int, default=1000,
        help="Evaluate every N steps",
    )
    output_group.add_argument(
        "--save-strategy", type=str, default="steps", choices=["steps", "epoch", "no"],
        help="Save strategy",
    )
    output_group.add_argument(
        "--save-steps", type=int, default=5000,
        help="Save checkpoint every N steps",
    )
    output_group.add_argument(
        "--save-total-limit", type=int, default=2,
        help="Max checkpoints to keep",
    )
    output_group.add_argument(
        "--no-load-best", action="store_true",
        help="Don't load best model at end",
    )

    # Resume
    p.add_argument(
        "--resume", action="store_true", dest="resume_from_checkpoint",
        help="Resume from latest checkpoint",
    )

    # Logging settings
    log_group = p.add_argument_group("Logging")
    log_group.add_argument(
        "--no-wandb", action="store_true",
        help="Disable wandb logging",
    )
    log_group.add_argument(
        "--wandb-project", type=str, default="semantic-id-full-finetune",
        help="Wandb project name",
    )
    log_group.add_argument(
        "--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"],
        help="Wandb mode",
    )

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Точка входа для full fine-tuning."""
    args = parse_args(argv)

    from .config import FullFineTuneConfig
    from .model import load_model_for_full_finetune
    from .train import finetune_model, finish_wandb, save_final_model

    # Создаём конфигурацию
    config = FullFineTuneConfig(
        stage1_checkpoint=args.stage1_checkpoint,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        gradient_checkpointing=args.gradient_checkpointing,
        random_state=args.random_state,
        num_proc=args.num_proc,
        category=args.category,
        data_dir=args.data_dir,
        use_full_dataset=args.use_full_dataset,
        max_training_samples=args.max_training_samples,
        eval_samples=args.eval_samples,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clip_norm=args.gradient_clip_norm,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=not args.no_load_best,
        resume_from_checkpoint=args.resume_from_checkpoint,
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

    # Обучаем
    result = finetune_model(config)

    # Загружаем модель для сохранения финальной версии
    model, tokenizer = load_model_for_full_finetune(config)

    # Сохраняем финальную модель
    final_path = save_final_model(model, tokenizer, config, result)

    # Завершаем wandb
    finish_wandb()

    logger.info("=" * 50)
    logger.info("Stage 2: Full fine-tuning complete!")
    logger.info(f"Training loss: {result.training_loss}")
    logger.info(f"Total steps: {result.global_step}")
    logger.info(f"Runtime: {result.runtime_seconds:.1f}s")
    logger.info(f"Final model: {final_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

