"""Слои векторного квантования для RQ-VAE.

Содержит:
- QuantizationOutput: structured output квантователя
- BaseVectorQuantizer: базовый класс с общей логикой
- VectorQuantizer: обычный VQ с градиентным обновлением кодбука
- EMAVectorQuantizer: VQ с EMA-обновлением кодбука
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import RQVAEConfig

logger = logging.getLogger("train-rqvae")


@dataclass
class QuantizationOutput:
    """Результат квантования одного VQ-слоя.

    Attributes:
        quantized_st: Квантованный вектор с градиентами (через STE/rotation trick).
                      Используется для backprop.
        quantized: Квантованный вектор без градиентов (чистый lookup из кодбука).
                   Используется для вычисления residual.
        indices: Индексы выбранных кодов ``[B]`` или ``[B, ...]``.
        loss: Суммарный VQ-loss (codebook + commitment).
        codebook_loss: MSE между входом и квантованным (учит кодбук).
                       None для EMA-квантователя.
        commitment_loss: MSE между входом и квантованным (учит encoder "прилипать").
    """

    quantized_st: Tensor
    quantized: Tensor
    indices: Tensor
    loss: Tensor
    codebook_loss: Optional[Tensor]
    commitment_loss: Tensor


class BaseVectorQuantizer(nn.Module):
    """Базовый класс для VQ-слоя.

    Содержит общую логику:
    - Кодбук (nn.Embedding)
    - Поиск ближайшего кода (оптимизированный)
    - STE / rotation trick для градиентов
    - Статистика использования кодбука
    """

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

    # -------------------------------------------------------------------------
    # Rotation trick (https://arxiv.org/abs/2410.06424)
    # -------------------------------------------------------------------------

    @staticmethod
    def l2norm(t: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
        """L2-нормализация."""
        return F.normalize(t, p=2, dim=dim, eps=eps)

    @staticmethod
    def safe_div(num: Tensor, den: Tensor, eps: float = 1e-6) -> Tensor:
        """Безопасное деление."""
        return num / den.clamp(min=eps)

    @staticmethod
    def rotation_trick(u: Tensor, q: Tensor, e: Tensor) -> Tensor:
        """Rotation trick (Eq 4.2 из статьи)."""
        w = BaseVectorQuantizer.l2norm(u + q, dim=-1).detach()
        w_col = w.unsqueeze(-1)
        w_row = w.unsqueeze(-2)
        u_col = u.unsqueeze(-1).detach()
        q_row = q.unsqueeze(-2).detach()

        if e.ndim == 2:
            e_expanded = e.unsqueeze(1)
            result = e_expanded - 2 * (e_expanded @ w_col @ w_row) + 2 * (e_expanded @ u_col @ q_row)
            return result.squeeze(1)
        return e - 2 * (e @ w_col @ w_row).squeeze(-1) + 2 * (e @ u_col @ q_row).squeeze(-1)

    @staticmethod
    def rotate_to(src: Tensor, tgt: Tensor) -> Tensor:
        """STE через rotation trick: forward = tgt, градиенты ≈ через src."""
        orig_shape = src.shape
        src_flat = src.reshape(-1, src.shape[-1])
        tgt_flat = tgt.reshape(-1, tgt.shape[-1])

        norm_src = src_flat.norm(dim=-1, keepdim=True)
        norm_tgt = tgt_flat.norm(dim=-1, keepdim=True)

        rotated_tgt = BaseVectorQuantizer.rotation_trick(
            BaseVectorQuantizer.safe_div(src_flat, norm_src),
            BaseVectorQuantizer.safe_div(tgt_flat, norm_tgt),
            src_flat,
        )
        rotated = rotated_tgt * BaseVectorQuantizer.safe_div(norm_tgt, norm_src).detach()
        return rotated.reshape(orig_shape)

    # -------------------------------------------------------------------------
    # Core quantization
    # -------------------------------------------------------------------------

    def find_nearest_codes(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Найти ближайшие коды в кодбуке.

        Использует decomposition: ||x - e||² = ||x||² + ||e||² - 2⟨x, e⟩
        Это быстрее torch.cdist для больших кодбуков.

        Args:
            x: Входной тензор ``[B, D]`` или ``[B, ..., D]``.

        Returns:
            Кортеж ``(indices, quantized)``:
            - indices: индексы ближайших кодов
            - quantized: квантованные векторы из кодбука
        """
        input_shape = x.shape
        flat_x = x.reshape(-1, self.codebook_embedding_dim)

        # ||x - e||² = ||x||² + ||e||² - 2⟨x, e⟩
        # Это эффективнее cdist для больших K
        x_sq = (flat_x ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        e_sq = (self.embedding.weight ** 2).sum(dim=1)  # [K]
        xe = flat_x @ self.embedding.weight.T  # [B, K]

        distances = x_sq + e_sq - 2 * xe  # [B, K]
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view(input_shape)

        return indices.view(input_shape[:-1]), quantized

    def apply_gradient_estimator(self, x: Tensor, quantized: Tensor) -> Tensor:
        """Применить STE или rotation trick для проброса градиентов."""
        if self.training and x.requires_grad:
            if self.use_rotation_trick:
                return self.rotate_to(x, quantized)
            # Straight-through estimator
            return x + (quantized - x).detach()
        return quantized

    # -------------------------------------------------------------------------
    # Usage tracking
    # -------------------------------------------------------------------------

    def update_usage(self, indices: Tensor) -> None:
        """Обновить статистику использования кодов."""
        indices_flat = indices.flatten()
        self.usage_count.scatter_add_(0, indices_flat, torch.ones_like(indices_flat, dtype=torch.float))
        self.update_count += 1

    def get_usage_rate(self) -> float:
        """Доля кодов, которые были использованы хотя бы раз."""
        if self.update_count == 0:
            return 0.0
        return (self.usage_count > 0).float().mean().item()

    def reset_usage_count(self) -> None:
        """Сбросить счётчик использования."""
        self.usage_count.zero_()


class VectorQuantizer(BaseVectorQuantizer):
    """Обычный VQ: кодбук учится градиентом через codebook_loss."""

    def forward(self, x: Tensor) -> QuantizationOutput:
        """Forward pass.

        Args:
            x: Входной тензор ``[B, D]``.

        Returns:
            QuantizationOutput с квантованными векторами, индексами и лоссами.
        """
        indices, quantized = self.find_nearest_codes(x)
        quantized_st = self.apply_gradient_estimator(x, quantized)

        # Losses
        codebook_loss = F.mse_loss(x.detach(), quantized)  # учит кодбук
        commitment_loss = F.mse_loss(x, quantized.detach())  # учит encoder
        loss = codebook_loss + self.commitment_weight * commitment_loss

        if self.training:
            self.update_usage(indices)

        return QuantizationOutput(quantized_st, quantized, indices, loss, codebook_loss, commitment_loss)

    def reset_unused_codes(self, batch_data: Tensor) -> None:
        """Сбросить неиспользуемые коды на случайные векторы из батча."""
        if self.update_count == 0:
            return

        unused_indices = (self.usage_count == 0).nonzero().squeeze(-1)
        if len(unused_indices) > 0 and batch_data.shape[0] >= len(unused_indices):
            batch_flat = batch_data.reshape(-1, self.codebook_embedding_dim)
            random_indices = torch.randperm(batch_flat.shape[0], device=batch_flat.device)[: len(unused_indices)]
            self.embedding.weight.data[unused_indices] = batch_flat[random_indices].detach()

        self.reset_usage_count()


class EMAVectorQuantizer(BaseVectorQuantizer):
    """VQ с EMA-обновлением кодбука (без codebook_loss)."""

    def __init__(self, config: RQVAEConfig) -> None:
        super().__init__(config)
        self.decay = config.ema_decay
        self.epsilon = config.ema_epsilon

        self.register_buffer("ema_cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    @staticmethod
    def ema_inplace(moving_avg: Tensor, new: Tensor, decay: float) -> None:
        """In-place EMA update."""
        moving_avg.data.mul_(decay).add_(new, alpha=1 - decay)

    @staticmethod
    def laplace_smoothing(x: Tensor, n_categories: int, epsilon: float = 1e-5) -> Tensor:
        """Laplace smoothing для численной стабильности."""
        return (x + epsilon) / (x.sum() + n_categories * epsilon) * x.sum()

    def forward(self, x: Tensor) -> QuantizationOutput:
        """Forward pass с EMA-обновлением кодбука.

        Args:
            x: Входной тензор ``[B, D]``.

        Returns:
            QuantizationOutput (codebook_loss = None).
        """
        indices, quantized = self.find_nearest_codes(x)

        if self.training:
            flat_x = x.reshape(-1, self.codebook_embedding_dim)
            flat_indices = indices.flatten()

            # EMA update
            encodings = F.one_hot(flat_indices, self.codebook_size).float()
            self.ema_inplace(self.ema_cluster_size, encodings.sum(0), self.decay)
            dw = encodings.T @ flat_x
            self.ema_inplace(self.ema_w, dw, self.decay)

            # Update codebook
            cluster_size = self.laplace_smoothing(self.ema_cluster_size, self.codebook_size, self.epsilon)
            self.embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)

            self.update_usage(indices)

        quantized_st = self.apply_gradient_estimator(x, quantized)

        # Только commitment loss (кодбук учится через EMA)
        commitment_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_weight * commitment_loss

        return QuantizationOutput(quantized_st, quantized, indices, loss, None, commitment_loss)
