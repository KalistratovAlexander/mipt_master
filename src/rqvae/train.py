"""Тренировочный цикл RQ-VAE."""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from .config import RQVAEConfig
from .metrics import avg_residual_norm, codebook_usage, unique_ids_proportion
from .model import ForwardOutput, RQVAE
from .quantization import EMAVectorQuantizer, VectorQuantizer

logger = logging.getLogger("train-rqvae")


@dataclass
class TrainState:
    """Состояние обучения (для resume)."""

    epoch: int = 0  # 0-indexed
    global_step: int = 0
    best_loss: float = float("inf")
    optimizer_state_dict: Optional[dict[str, Any]] = None
    scheduler_state_dict: Optional[dict[str, Any]] = None


def get_gradient_norm(model: torch.nn.Module) -> float:
    """L2-норма градиентов по всем параметрам."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
    return total_norm.item()


@torch.no_grad()
def get_loss(model: RQVAE, val_loader: DataLoader, device: str) -> float:
    """Средний loss по валидационному датасету."""
    model.eval()
    total_loss = 0.0
    batch_count = 0
    for data in val_loader:
        if isinstance(data, (list, tuple)):
            data = data[0]
        data = data.to(device)
        output: ForwardOutput = model(data)
        total_loss += float(output.loss.item())
        batch_count += 1
    return total_loss / max(batch_count, 1)


def evaluate(
    model: RQVAE,
    val_loader: DataLoader,
    train_loader: DataLoader,
    device: str,
    global_step: int,
    epoch: int,  # 1-indexed для логов
) -> dict:
    """Валидация: val_loss + метрики на sample batch."""
    model.eval()

    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        if isinstance(sample_batch, (list, tuple)):
            sample_batch = sample_batch[0]
        sample_batch = sample_batch.to(device)

        output: ForwardOutput = model(sample_batch)

        usage = codebook_usage(model.vq_layers)
        residual_norm = avg_residual_norm(output.residual)
        semantic_ids = torch.stack(output.indices, dim=-1)
        unique_proportion = unique_ids_proportion(semantic_ids)

    val_loss = get_loss(model, val_loader, device)
    usage_str = "/".join([f"{u:.2f}" for u in usage])

    logger.info(
        f"Step {global_step:05d} | Epoch {epoch:05d} | Val loss: {val_loss:.2e} | "
        f"Codebook usage: {usage_str} | Avg residual norm: {residual_norm:.3f} | "
        f"Unique ids: {unique_proportion:.1%}"
    )

    return {
        "val_loss": val_loss,
        "codebook_usage": usage,
        "codebook_usage_str": usage_str,
        "avg_residual_norm": residual_norm,
        "unique_ids_proportion": unique_proportion,
    }


def save_checkpoint(
    model: RQVAE,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    metrics: dict,
    config: RQVAEConfig,
    global_step: int,
    epoch: int,
    best_loss: float,
) -> float:
    """Сохранить чекпоинт и обновить best_loss."""
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "epoch": epoch,
        "step": global_step,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_loss": metrics["val_loss"],
        "config": config.__dict__,
    }

    checkpoint_path = config.checkpoint_dir / f"checkpoint_step_{global_step}.pth"
    torch.save(checkpoint_data, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    if metrics["val_loss"] < best_loss:
        best_loss = metrics["val_loss"]
        best_model_path = config.checkpoint_dir / "best_model.pth"
        checkpoint_data["best_loss"] = best_loss
        torch.save(checkpoint_data, best_model_path)
        logger.info(f"Saved best model with val_loss: {best_loss:.4e}")

    return best_loss


def train_rqvae(
    model: RQVAE,
    data_loader: DataLoader,
    config: RQVAEConfig,
    device: str = "cpu",
    val_loader: Optional[DataLoader] = None,
    state: Optional[TrainState] = None,
) -> TrainState:
    """Тренировочный цикл. Возвращает состояние (epoch/step/best_loss) для resume."""
    model = model.to(device)
    state = state or TrainState()

    if config.use_kmeans_init and state.global_step == 0 and state.epoch == 0:
        model.kmeans_init(data_loader, device)

    if device == "cuda":
        logger.info("Compiling model with torch.compile for faster training...")
        model = torch.compile(model)

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr, weight_decay=0.01, fused=use_fused)

    # Scheduler
    steps_per_epoch = max(1, len(data_loader) // config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.num_epochs
    logger.info(f"Total training steps: {total_steps:,} ({steps_per_epoch} steps/epoch x {config.num_epochs} epochs)")

    if config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=config.min_lr)
        logger.info(f"Cosine annealing: {config.max_lr:.1e} -> {config.min_lr:.1e} for {total_steps:,} steps")
    elif config.scheduler_type == "cosine_with_warmup":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.warmup_start_lr / config.max_lr,
            total_iters=config.warmup_steps,
        )
        logger.info(f"Warmup: {config.warmup_start_lr:.1e} -> {config.max_lr:.1e} for {config.warmup_steps:,} steps")

        cosine_steps = max(1, total_steps - config.warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=config.min_lr)
        logger.info(f"Cosine annealing: {config.max_lr:.1e} -> {config.min_lr:.1e} for {cosine_steps:,} steps")

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_steps])
    else:
        scheduler = None

    # Resume states
    if state.optimizer_state_dict is not None:
        optimizer.load_state_dict(state.optimizer_state_dict)
    if scheduler is not None and state.scheduler_state_dict is not None:
        scheduler.load_state_dict(state.scheduler_state_dict)

    best_loss = state.best_loss
    global_step = state.global_step

    for epoch in range(state.epoch, config.num_epochs):
        model.train()

        for batch_idx, data in enumerate(data_loader):
            if batch_idx % config.gradient_accumulation_steps == 0:
                t0 = time.time()
                optimizer.zero_grad()
                loss_accum = 0.0

            if isinstance(data, (list, tuple)):
                data = data[0]

            output: ForwardOutput = model(data.to(device))
            loss = output.loss / config.gradient_accumulation_steps
            loss_accum += float(output.loss.detach().item())

            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                grad_norm_before = get_gradient_norm(model)

                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
                    grad_norm_after = get_gradient_norm(model)
                else:
                    grad_norm_after = grad_norm_before

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                t1 = time.time()
                batch_time_ms = (t1 - t0) * 1000
                samples_per_second = (data.shape[0] * config.gradient_accumulation_steps) / max((t1 - t0), 1e-9)
                global_step += 1

                avg_loss = loss_accum / config.gradient_accumulation_steps

                if global_step == 1 or global_step % config.steps_per_train_log == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    usage = codebook_usage(model.vq_layers)
                    semantic_ids = torch.stack(output.indices, dim=-1)
                    unique_proportion = unique_ids_proportion(semantic_ids)
                    usage_str = "/".join([f"{u:.2f}" for u in usage])

                    logger.info(
                        f"Step {global_step:05d} | Epoch {epoch + 1:05d} | lr: {current_lr:.2e} | "
                        f"loss: {avg_loss:.2e} | recon: {output.recon_loss.item():.2e} | "
                        f"vq: {output.vq_loss.item():.2e} | codebook usage: {usage_str} | "
                        f"unique ids: {unique_proportion:.1%} | time: {batch_time_ms:.0f}ms | "
                        f"samples/s: {samples_per_second:,.0f}"
                    )

                if global_step % config.steps_per_val_log == 0 and val_loader is not None:
                    metrics = evaluate(model, val_loader, data_loader, device, global_step, epoch + 1)
                    best_loss = save_checkpoint(
                        model, optimizer, scheduler, metrics, config, global_step, epoch, best_loss
                    )
                    model.train()

                if config.reset_unused_codes and global_step % config.steps_per_codebook_reset == 0:
                    if config.scheduler_type == "cosine_with_warmup" and global_step < config.warmup_steps:
                        logger.debug(f"Step {global_step:05d} - Skipping codebook reset during warmup")
                    else:
                        model.eval()
                        usage = codebook_usage(model.vq_layers)

                        reset_batch = next(iter(data_loader))
                        if isinstance(reset_batch, (list, tuple)):
                            reset_batch = reset_batch[0]

                        for level, vq_layer in enumerate(model.vq_layers):
                            if isinstance(vq_layer, VectorQuantizer) and not isinstance(vq_layer, EMAVectorQuantizer):
                                if usage[level] < config.codebook_usage_threshold:
                                    with torch.no_grad():
                                        z = model.encode(reset_batch.to(device))
                                        residual = z
                                        for i in range(level):
                                            vq_out = model.vq_layers[i](residual)
                                            residual = residual - vq_out.quantized
                                        vq_layer.reset_unused_codes(residual)
                                vq_layer.reset_usage_count()
                        model.train()

        # неполная аккумуляция в конце эпохи
        if (batch_idx + 1) % config.gradient_accumulation_steps != 0:
            if config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

        state.epoch = epoch + 1
        state.global_step = global_step
        state.best_loss = best_loss

    return state
