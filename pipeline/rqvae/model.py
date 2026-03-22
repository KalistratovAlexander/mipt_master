"""RQ-VAE: config, vector quantization layers, and model.

Architecture: Encoder MLP → Residual VQ (L levels) → Decoder MLP.
Each item embedding gets a semantic ID = tuple of codebook indices per level.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("train-rqvae")


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class RQVAEConfig:
    # Data
    embeddings_path: Path = field(default_factory=lambda: Path("data/embeds/Pet_Supplies_items_with_embeddings.parquet"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("models/rqvae"))

    # Model
    item_embedding_dim: int = 1024
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    codebook_embedding_dim: int = 32
    codebook_quantization_levels: int = 3
    codebook_size: int = 256
    commitment_weight: float = 0.25
    use_rotation_trick: bool = True

    # EMA VQ (alternative to gradient-based VQ)
    use_ema_vq: bool = False
    ema_decay: float = 0.99
    ema_epsilon: float = 1e-5

    # Training
    batch_size: int = 4096
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1000
    scheduler_type: str = "cosine_with_warmup"
    warmup_start_lr: float = 1e-8
    warmup_steps: int = 200
    max_lr: float = 3e-4
    min_lr: float = 1e-6
    use_gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0
    use_kmeans_init: bool = True
    reset_unused_codes: bool = True
    steps_per_codebook_reset: int = 2
    codebook_usage_threshold: float = 1.0
    val_split: float = 0.05

    # Logging
    steps_per_train_log: int = 10
    steps_per_val_log: int = 200
    seed: int = 42

    def validate(self) -> None:
        assert self.embeddings_path is not None, "embeddings_path is required"
        assert self.batch_size > 0
        assert self.gradient_accumulation_steps > 0
        assert self.num_epochs > 0
        assert self.codebook_quantization_levels > 0
        assert self.codebook_size > 0
        assert self.codebook_embedding_dim > 0
        assert 0.0 <= self.val_split < 1.0
        assert self.scheduler_type in {"cosine", "cosine_with_warmup", "none"}

    def log_config(self) -> None:
        eff = self.batch_size * self.gradient_accumulation_steps
        lines = [
            f"=== RQ-VAE Config ===",
            f"  data: {self.embeddings_path}",
            f"  model: dim={self.item_embedding_dim} -> enc{self.encoder_hidden_dims} -> "
            f"codebook(K={self.codebook_size}, D={self.codebook_embedding_dim}, L={self.codebook_quantization_levels})",
            f"  vq: {'EMA' if self.use_ema_vq else 'gradient'}, "
            f"rotation_trick={self.use_rotation_trick}, commitment={self.commitment_weight}",
            f"  training: epochs={self.num_epochs}, batch={self.batch_size}x{self.gradient_accumulation_steps}={eff}",
            f"  lr: {self.max_lr:.1e} -> {self.min_lr:.1e} ({self.scheduler_type}), "
            f"warmup={self.warmup_steps} steps",
            f"  checkpoints: {self.checkpoint_dir}",
        ]
        logger.info("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════
# Vector Quantization
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QuantizationOutput:
    quantized_st: Tensor   # with gradients (STE / rotation trick)
    quantized: Tensor      # without gradients (for residual computation)
    indices: Tensor        # codebook indices [B]
    loss: Tensor           # total VQ loss
    codebook_loss: Optional[Tensor]
    commitment_loss: Tensor


class BaseVectorQuantizer(nn.Module):
    """Base VQ layer: codebook, nearest-code lookup, STE/rotation trick, usage tracking."""

    def __init__(self, config: RQVAEConfig) -> None:
        super().__init__()
        self.codebook_embedding_dim = config.codebook_embedding_dim
        self.codebook_size = config.codebook_size
        self.commitment_weight = config.commitment_weight
        self.use_rotation_trick = config.use_rotation_trick

        self.embedding = nn.Embedding(self.codebook_size, self.codebook_embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.codebook_size, 1 / self.codebook_size)

        self.register_buffer("usage_count", torch.zeros(self.codebook_size))
        self.register_buffer("update_count", torch.tensor(0))

    # --- Rotation trick (https://arxiv.org/abs/2410.06424) ---

    @staticmethod
    def _l2norm(t: Tensor, eps: float = 1e-6) -> Tensor:
        return F.normalize(t, p=2, dim=-1, eps=eps)

    @staticmethod
    def _safe_div(num: Tensor, den: Tensor, eps: float = 1e-6) -> Tensor:
        return num / den.clamp(min=eps)

    @staticmethod
    def _rotation_trick(u: Tensor, q: Tensor, e: Tensor) -> Tensor:
        w = BaseVectorQuantizer._l2norm(u + q).detach()
        w_col, w_row = w.unsqueeze(-1), w.unsqueeze(-2)
        u_col = u.unsqueeze(-1).detach()
        q_row = q.unsqueeze(-2).detach()
        if e.ndim == 2:
            e = e.unsqueeze(1)
            return (e - 2 * (e @ w_col @ w_row) + 2 * (e @ u_col @ q_row)).squeeze(1)
        return e - 2 * (e @ w_col @ w_row).squeeze(-1) + 2 * (e @ u_col @ q_row).squeeze(-1)

    @staticmethod
    def _rotate_to(src: Tensor, tgt: Tensor) -> Tensor:
        shape = src.shape
        s, t = src.reshape(-1, shape[-1]), tgt.reshape(-1, shape[-1])
        ns, nt = s.norm(dim=-1, keepdim=True), t.norm(dim=-1, keepdim=True)
        sd = BaseVectorQuantizer._safe_div
        rotated = BaseVectorQuantizer._rotation_trick(sd(s, ns), sd(t, nt), s)
        return (rotated * sd(nt, ns).detach()).reshape(shape)

    # --- Core ---

    def find_nearest_codes(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Find nearest codebook entries. Returns (indices, quantized)."""
        shape = x.shape
        flat = x.reshape(-1, self.codebook_embedding_dim)
        # ||x - e||² = ||x||² + ||e||² - 2⟨x, e⟩
        dist = (flat ** 2).sum(1, keepdim=True) + (self.embedding.weight ** 2).sum(1) - 2 * flat @ self.embedding.weight.T
        idx = dist.argmin(dim=1)
        return idx.view(shape[:-1]), self.embedding(idx).view(shape)

    def apply_gradient_estimator(self, x: Tensor, quantized: Tensor) -> Tensor:
        if self.training and x.requires_grad:
            return self._rotate_to(x, quantized) if self.use_rotation_trick else x + (quantized - x).detach()
        return quantized

    # --- Usage tracking ---

    def update_usage(self, indices: Tensor) -> None:
        self.usage_count.scatter_add_(0, indices.flatten(), torch.ones_like(indices.flatten(), dtype=torch.float))
        self.update_count += 1

    def get_usage_rate(self) -> float:
        return (self.usage_count > 0).float().mean().item() if self.update_count > 0 else 0.0

    def reset_usage_count(self) -> None:
        self.usage_count.zero_()


class VectorQuantizer(BaseVectorQuantizer):
    """Standard VQ: codebook learns via gradient through codebook_loss."""

    def forward(self, x: Tensor) -> QuantizationOutput:
        indices, quantized = self.find_nearest_codes(x)
        quantized_st = self.apply_gradient_estimator(x, quantized)
        codebook_loss = F.mse_loss(x.detach(), quantized)
        commitment_loss = F.mse_loss(x, quantized.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss
        if self.training:
            self.update_usage(indices)
        return QuantizationOutput(quantized_st, quantized, indices, loss, codebook_loss, commitment_loss)

    def reset_unused_codes(self, batch_data: Tensor) -> None:
        if self.update_count == 0:
            return
        unused = (self.usage_count == 0).nonzero().squeeze(-1)
        if len(unused) > 0 and batch_data.shape[0] >= len(unused):
            flat = batch_data.reshape(-1, self.codebook_embedding_dim)
            perm = torch.randperm(flat.shape[0], device=flat.device)[:len(unused)]
            self.embedding.weight.data[unused] = flat[perm].detach()
        self.reset_usage_count()


class EMAVectorQuantizer(BaseVectorQuantizer):
    """EMA VQ: codebook updates via exponential moving average (no codebook_loss)."""

    def __init__(self, config: RQVAEConfig) -> None:
        super().__init__(config)
        self.decay = config.ema_decay
        self.epsilon = config.ema_epsilon
        self.register_buffer("ema_cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, x: Tensor) -> QuantizationOutput:
        indices, quantized = self.find_nearest_codes(x)
        if self.training:
            flat = x.reshape(-1, self.codebook_embedding_dim)
            enc = F.one_hot(indices.flatten(), self.codebook_size).float()
            self.ema_cluster_size.mul_(self.decay).add_(enc.sum(0), alpha=1 - self.decay)
            self.ema_w.mul_(self.decay).add_(enc.T @ flat, alpha=1 - self.decay)
            n = (self.ema_cluster_size + self.epsilon) / (self.ema_cluster_size.sum() + self.codebook_size * self.epsilon) * self.ema_cluster_size.sum()
            self.embedding.weight.data = self.ema_w / n.unsqueeze(1)
            self.update_usage(indices)
        quantized_st = self.apply_gradient_estimator(x, quantized)
        commitment_loss = F.mse_loss(x, quantized.detach())
        return QuantizationOutput(quantized_st, quantized, indices, self.commitment_weight * commitment_loss, None, commitment_loss)


# ═══════════════════════════════════════════════════════════════════════════
# RQ-VAE Model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ForwardOutput:
    x_recon: Tensor
    indices: list[Tensor]
    loss: Tensor
    recon_loss: Tensor
    vq_loss: Tensor
    codebook_losses: list[Tensor]
    commitment_losses: list[Tensor]
    residual: Tensor


def _torch_kmeans(data: Tensor, k: int, n_iter: int = 10, seed: int = 0) -> Tensor:
    """GPU-friendly k-means. Returns centroids [K, D]."""
    n, d = data.shape
    gen = torch.Generator(device=data.device).manual_seed(seed)
    centroids = data[torch.randperm(n, generator=gen, device=data.device)[:k]].clone()

    for _ in range(n_iter):
        dist = (data ** 2).sum(1, keepdim=True) + (centroids ** 2).sum(1) - 2 * data @ centroids.T
        assign = dist.argmin(dim=1)
        new = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=data.device)
        for i in range(d):
            new[:, i].scatter_add_(0, assign, data[:, i])
        counts.scatter_add_(0, assign, torch.ones(n, device=data.device))
        mask = counts > 0
        new[mask] /= counts[mask].unsqueeze(1)
        new[~mask] = centroids[~mask]
        centroids = new
    return centroids


def _build_mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.SiLU()])
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class RQVAE(nn.Module):
    """Residual Quantized VAE for hierarchical semantic ID generation.

    Encoder MLP → L levels of residual vector quantization → Decoder MLP.
    """

    def __init__(self, config: RQVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.codebook_embedding_dim = config.codebook_embedding_dim
        self.codebook_size = config.codebook_size
        self.codebook_quantization_levels = config.codebook_quantization_levels

        enc_dims = [config.item_embedding_dim, *config.encoder_hidden_dims, config.codebook_embedding_dim]
        dec_dims = [config.codebook_embedding_dim, *config.encoder_hidden_dims[::-1], config.item_embedding_dim]
        self.encoder = _build_mlp(enc_dims)
        self.decoder = _build_mlp(dec_dims)

        vq_cls = EMAVectorQuantizer if config.use_ema_vq else VectorQuantizer
        self.vq_layers = nn.ModuleList([vq_cls(config) for _ in range(config.codebook_quantization_levels)])

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> ForwardOutput:
        z = self.encode(x)
        quantized_out = torch.zeros_like(z)
        residual = z
        indices, vq_loss = [], torch.tensor(0.0, device=x.device)
        cb_losses, cm_losses = [], []

        for vq in self.vq_layers:
            out = vq(residual)
            residual = residual - out.quantized.detach()
            quantized_out = quantized_out + out.quantized_st
            indices.append(out.indices)
            vq_loss = vq_loss + out.loss
            if out.codebook_loss is not None:
                cb_losses.append(out.codebook_loss)
            cm_losses.append(out.commitment_loss)

        x_recon = self.decode(quantized_out)
        recon_loss = F.mse_loss(x_recon, x)
        return ForwardOutput(x_recon, indices, recon_loss + vq_loss, recon_loss, vq_loss, cb_losses, cm_losses, residual)

    @torch.no_grad()
    def encode_to_semantic_ids(self, x: Tensor) -> Tensor:
        """Encode to semantic IDs [B, L] without gradients."""
        z = self.encode(x)
        residual, ids = z, []
        for vq in self.vq_layers:
            out = vq(residual)
            ids.append(out.indices)
            residual = residual - out.quantized
        return torch.stack(ids, dim=-1)

    @torch.no_grad()
    def decode_from_semantic_ids(self, semantic_ids: Tensor) -> Tensor:
        """Decode semantic IDs [B, L] back to embeddings."""
        q = torch.zeros(semantic_ids.shape[0], self.codebook_embedding_dim, device=semantic_ids.device)
        for lvl, idx in enumerate(semantic_ids.unbind(dim=-1)):
            q += self.vq_layers[lvl].embedding(idx)
        return self.decode(q)

    @torch.no_grad()
    def kmeans_init(self, data_loader, device: str) -> None:
        """Initialize codebooks with k-means on first batch."""
        logger.info("K-means codebook initialization...")
        batch = next(iter(data_loader))
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        z = self.encode(batch.to(device))
        residual = z
        for lvl, vq in enumerate(self.vq_layers):
            vq.embedding.weight.data = _torch_kmeans(residual, self.codebook_size, n_iter=10, seed=lvl)
            logger.info(f"  Level {lvl}: {self.codebook_size} codes initialized")
            if lvl < self.codebook_quantization_levels - 1:
                residual = residual - vq(residual).quantized
