"""RQ-VAE training: data loading, training loop, checkpointing.

Usage as CLI:
    python -m mipt_master.src.rqvae.train --embeddings-path data/embeds/Pet_Supplies_items_with_embeddings.parquet
"""

from __future__ import annotations

import argparse
import inspect
import logging
import re
import sys
import time
from dataclasses import dataclass
from multiprocessing import current_process
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .model import (
    EMAVectorQuantizer,
    ForwardOutput,
    RQVAE,
    RQVAEConfig,
    VectorQuantizer,
)

logger = logging.getLogger("train-rqvae")


# ═══════════════════════════════════════════════════════════════════════════
# Device management
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DeviceInfo:
    """Device capabilities for optimal DataLoader and training configuration."""
    device: str
    is_cuda: bool
    is_mps: bool
    is_cpu: bool
    supports_pin_memory: bool
    default_num_workers: int
    default_prefetch_factor: Optional[int]

    def get_dataloader_kwargs(self) -> dict:
        workers = self.default_num_workers
        prefetch = self.default_prefetch_factor
        kwargs = {"num_workers": workers, "pin_memory": self.supports_pin_memory}
        if workers > 0 and prefetch is not None:
            kwargs["prefetch_factor"] = prefetch
            kwargs["persistent_workers"] = True
        return kwargs


def get_device_info() -> DeviceInfo:
    """Detect best available device and return optimal configuration."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        return DeviceInfo("cuda", True, False, False, True, 16, 8)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return DeviceInfo("mps", False, True, False, False, 0, None)
    return DeviceInfo("cpu", False, False, True, False, 4, 2)


# ═══════════════════════════════════════════════════════════════════════════
# Logger setup
# ═══════════════════════════════════════════════════════════════════════════

def setup_logger(
    name: str = "train-rqvae", level: int = logging.INFO,
    log_to_file: bool = False, log_dir: str | Path = "logs",
) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(level)
    log.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(console)

    if log_to_file and current_process().name == "MainProcess":
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(d / f"log-{time.strftime('%Y%m%d_%H%M%S')}.txt", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)-5s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        log.addHandler(fh)
        log.info(f"Log file: {d / fh.baseFilename}")
    return log


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_embeddings(path: str | Path) -> Tensor:
    """Load embeddings from parquet via pyarrow (zero-copy where possible)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings not found: {path}")

    logger.info(f"Loading embeddings from {path}")
    table = pq.read_table(path, columns=["embedding"])
    if table.num_rows == 0:
        raise ValueError(f"Empty file: {path}")

    arr = table.column("embedding").combine_chunks()
    offsets = arr.offsets.to_numpy(zero_copy_only=False)
    d = int(offsets[1] - offsets[0])
    flat = arr.values.to_numpy(zero_copy_only=False)

    if flat.dtype != np.float32:
        flat = flat.astype(np.float32)
    elif not flat.flags.writeable:
        flat = flat.copy()

    emb = torch.from_numpy(flat.reshape(-1, d)).contiguous()
    logger.info(f"Loaded {len(emb):,} embeddings, dim={d}")
    return emb


def prepare_data(
    path: str | Path, config: RQVAEConfig, device_info: DeviceInfo,
) -> tuple[DataLoader, DataLoader]:
    """Load embeddings, split train/val, return DataLoaders."""
    emb = load_embeddings(path)
    n = len(emb)
    val_size = int(n * config.val_split)
    train_ds, val_ds = torch.utils.data.random_split(
        emb, [n - val_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    logger.info(f"Train: {n - val_size:,}, Val: {val_size:,}")

    kw = device_info.get_dataloader_kwargs()
    val_kw = {"num_workers": 0, "pin_memory": False} if device_info.is_mps else kw

    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, **kw),
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, **val_kw),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Training metrics
# ═══════════════════════════════════════════════════════════════════════════

def codebook_usage(vq_layers) -> list[float]:
    return [vq.get_usage_rate() for vq in vq_layers]


def unique_ids_proportion(sids: Tensor) -> float:
    B = sids.shape[0]
    if B <= 1:
        return 1.0
    matches = (sids.unsqueeze(1) == sids.unsqueeze(0)).all(dim=-1)
    return float((~torch.triu(matches, diagonal=1).any(dim=1)).sum().item() / B)


# ═══════════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Checkpoint:
    path: Path
    epoch: int
    step: int
    best_loss: float
    model_state_dict: dict[str, Any]
    optimizer_state_dict: Optional[dict[str, Any]]
    scheduler_state_dict: Optional[dict[str, Any]]
    config: dict[str, Any]


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Checkpoint:
    raw = torch.load(path, map_location=map_location, weights_only=False)
    return Checkpoint(
        path=path, epoch=int(raw.get("epoch", 0)), step=int(raw.get("step", 0)),
        best_loss=float(raw.get("best_loss", raw.get("val_loss", float("inf")))),
        model_state_dict=raw["model_state_dict"],
        optimizer_state_dict=raw.get("optimizer_state_dict"),
        scheduler_state_dict=raw.get("scheduler_state_dict"),
        config=raw.get("config", {}),
    )


_CKPT_RE = re.compile(r"checkpoint_step_(\d+)\.pth$")
_MAX_CHECKPOINTS = 3


def _save_checkpoint(
    model, optimizer, scheduler, metrics: dict, config: RQVAEConfig,
    step: int, epoch: int, best_loss: float,
) -> float:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "epoch": epoch, "step": step, "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_loss": metrics["val_loss"], "config": config.__dict__,
    }
    torch.save(data, config.checkpoint_dir / f"checkpoint_step_{step}.pth")

    existing = sorted(
        config.checkpoint_dir.glob("checkpoint_step_*.pth"),
        key=lambda p: int(m.group(1)) if (m := _CKPT_RE.search(p.name)) else 0,
    )
    while len(existing) > _MAX_CHECKPOINTS:
        existing.pop(0).unlink()

    if metrics["val_loss"] < best_loss:
        best_loss = metrics["val_loss"]
        data["best_loss"] = best_loss
        torch.save(data, config.checkpoint_dir / "best_model.pth")
        logger.info(f"New best model: val_loss={best_loss:.4e}")
    return best_loss


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float("inf")
    optimizer_state_dict: Optional[dict[str, Any]] = None
    scheduler_state_dict: Optional[dict[str, Any]] = None


def _build_scheduler(opt, cfg: RQVAEConfig, total: int):
    if cfg.scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=cfg.min_lr)
    if cfg.scheduler_type == "cosine_with_warmup":
        warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=cfg.warmup_start_lr / cfg.max_lr, total_iters=cfg.warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total - cfg.warmup_steps), eta_min=cfg.min_lr)
        return torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[cfg.warmup_steps])
    return None


@torch.no_grad()
def _val_loss(model, val_loader, device):
    model.eval()
    total, n = 0.0, 0
    for data in val_loader:
        if isinstance(data, (list, tuple)):
            data = data[0]
        total += model(data.to(device)).loss.item()
        n += 1
    return total / max(n, 1)


def _evaluate(model, val_loader, train_loader, device, step, epoch):
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        out: ForwardOutput = model(batch.to(device))
        usage = codebook_usage(model.vq_layers)
        sids = torch.stack(out.indices, dim=-1)
    val = _val_loss(model, val_loader, device)
    logger.info(
        f"Step {step:05d} | Epoch {epoch:05d} | Val: {val:.2e} | "
        f"Codebook: {'/'.join(f'{u:.2f}' for u in usage)} | "
        f"Residual: {out.residual.norm(dim=-1).mean().item():.3f} | "
        f"Unique: {unique_ids_proportion(sids):.1%}"
    )
    return {"val_loss": val}


def _reset_codebooks(model, data_loader, device, config):
    model.eval()
    usage = codebook_usage(model.vq_layers)
    batch = next(iter(data_loader))
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    for lvl, vq in enumerate(model.vq_layers):
        if isinstance(vq, VectorQuantizer) and not isinstance(vq, EMAVectorQuantizer):
            if usage[lvl] < config.codebook_usage_threshold:
                with torch.no_grad():
                    residual = model.encode(batch.to(device))
                    for i in range(lvl):
                        residual = residual - model.vq_layers[i](residual).quantized
                    vq.reset_unused_codes(residual)
            vq.reset_usage_count()
    model.train()


def train_rqvae(
    model: RQVAE, data_loader: DataLoader, config: RQVAEConfig,
    device: str = "cpu", val_loader: Optional[DataLoader] = None,
    state: Optional[TrainState] = None,
) -> TrainState:
    model = model.to(device)
    state = state or TrainState()

    if config.use_kmeans_init and state.global_step == 0:
        model.kmeans_init(data_loader, device)

    if device == "cuda":
        model = torch.compile(model)

    fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=config.max_lr, weight_decay=0.01, fused=fused)

    steps_per_epoch = max(1, len(data_loader) // config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.num_epochs
    sched = _build_scheduler(opt, config, total_steps)
    logger.info(f"Training: {total_steps:,} steps ({steps_per_epoch}/epoch x {config.num_epochs})")

    if state.optimizer_state_dict:
        opt.load_state_dict(state.optimizer_state_dict)
    if sched and state.scheduler_state_dict:
        sched.load_state_dict(state.scheduler_state_dict)

    best, step, ga = state.best_loss, state.global_step, config.gradient_accumulation_steps

    for epoch in range(state.epoch, config.num_epochs):
        model.train()
        for bi, data in enumerate(data_loader):
            if bi % ga == 0:
                t0, loss_acc = time.time(), 0.0
                opt.zero_grad()
            if isinstance(data, (list, tuple)):
                data = data[0]
            out: ForwardOutput = model(data.to(device))
            (out.loss / ga).backward()
            loss_acc += out.loss.item()

            if (bi + 1) % ga == 0:
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                opt.step()
                if sched:
                    sched.step()
                step += 1
                dt = time.time() - t0

                if step == 1 or step % config.steps_per_train_log == 0:
                    usage = codebook_usage(model.vq_layers)
                    sids = torch.stack(out.indices, dim=-1)
                    logger.info(
                        f"Step {step:05d} | Epoch {epoch+1:05d} | lr: {opt.param_groups[0]['lr']:.2e} | "
                        f"loss: {loss_acc/ga:.2e} | recon: {out.recon_loss.item():.2e} | "
                        f"vq: {out.vq_loss.item():.2e} | cb: {'/'.join(f'{u:.2f}' for u in usage)} | "
                        f"uniq: {unique_ids_proportion(sids):.1%} | {dt*1000:.0f}ms"
                    )

                if step % config.steps_per_val_log == 0 and val_loader:
                    metrics = _evaluate(model, val_loader, data_loader, device, step, epoch + 1)
                    best = _save_checkpoint(model, opt, sched, metrics, config, step, epoch, best)
                    model.train()

                if config.reset_unused_codes and step % config.steps_per_codebook_reset == 0:
                    if not (config.scheduler_type == "cosine_with_warmup" and step < config.warmup_steps):
                        _reset_codebooks(model, data_loader, device, config)

        # Flush incomplete accumulation
        if (bi + 1) % ga != 0:
            if config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            opt.step()
            opt.zero_grad()
            if sched:
                sched.step()
            step += 1

        state.epoch, state.global_step, state.best_loss = epoch + 1, step, best
    return state


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _cli_main(argv=None):
    from .model import set_seed

    setup_logger("train-rqvae", log_to_file=True)

    p = argparse.ArgumentParser(description="Train RQ-VAE")
    p.add_argument("--embeddings-path", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("models/rqvae"))
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path, default=None)
    args = p.parse_args(argv)

    cfg = RQVAEConfig(
        embeddings_path=args.embeddings_path, checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size, num_epochs=args.epochs, seed=args.seed,
    )
    cfg.validate()
    set_seed(cfg.seed)

    dev = get_device_info()
    cfg.log_config()
    train_loader, val_loader = prepare_data(cfg.embeddings_path, cfg, dev)
    model = RQVAE(cfg)

    state = TrainState()
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        model.load_state_dict(ckpt.model_state_dict)
        state = TrainState(ckpt.epoch, ckpt.step, ckpt.best_loss, ckpt.optimizer_state_dict, ckpt.scheduler_state_dict)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = train_rqvae(model=model, data_loader=train_loader, val_loader=val_loader, config=cfg, device=dev.device, state=state)

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
        "epoch": state.epoch, "step": state.global_step, "best_loss": state.best_loss,
    }, cfg.checkpoint_dir / "final_model.pth")


if __name__ == "__main__":
    _cli_main()
