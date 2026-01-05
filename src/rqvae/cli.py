"""CLI для обучения RQ-VAE."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from mipt_master.src.device import get_device_info
from mipt_master.src.logger import setup_logger

from .config import RQVAEConfig
from .data import prepare_data
from .model import RQVAE
from .train import TrainState, train_rqvae
from .utils import load_checkpoint, set_seed

logger = setup_logger("train-rqvae", log_to_file=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    p = argparse.ArgumentParser(description="Train RQ-VAE for semantic ID generation")
    p.add_argument("--category", type=str, default="Amazon_Fashion")
    p.add_argument("--data-dir", type=Path, default=Path("amazon/data"))
    p.add_argument("--embeddings-path", type=Path, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("amazon/checkpoints") / "rqvae")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--val-split", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path, default=None, help="Path to checkpoint_step_*.pth")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Точка входа для обучения RQ-VAE."""
    args = parse_args(argv)

    config = RQVAEConfig(
        category=args.category,
        data_dir=args.data_dir,
        embeddings_path=args.embeddings_path,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        val_split=args.val_split,
        seed=args.seed,
    )
    config.validate()

    set_seed(config.seed)

    device_info = get_device_info()
    device = device_info.device
    logger.info(f"Using device: {device}")

    config.log_config()

    # Вся работа с данными — в одном вызове
    train_loader, val_loader = prepare_data(
        path=config.embeddings_path,
        config=config,
        device_info=device_info,
    )

    model = RQVAE(config)

    state = TrainState()
    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        logger.info(f"Resuming from checkpoint: {ckpt.path} (epoch={ckpt.epoch}, step={ckpt.step})")
        model.load_state_dict(ckpt.model_state_dict)
        state.epoch = ckpt.epoch
        state.global_step = ckpt.step
        state.best_loss = ckpt.best_loss
        state.optimizer_state_dict = ckpt.optimizer_state_dict  # type: ignore[attr-defined]
        state.scheduler_state_dict = ckpt.scheduler_state_dict  # type: ignore[attr-defined]

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = train_rqvae(
        model=model,
        data_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        state=state,
    )

    final_path = config.checkpoint_dir / "final_model.pth"
    logger.info(f"Saving final model to {final_path}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in config.__dict__.items()},
            "epoch": state.epoch,
            "step": state.global_step,
            "best_loss": state.best_loss,
        },
        final_path,
    )
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
