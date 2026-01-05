"""Модель RQ-VAE для генерации семантических ID."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import RQVAEConfig
from .quantization import EMAVectorQuantizer, VectorQuantizer

logger = logging.getLogger("train-rqvae")


@dataclass
class ForwardOutput:
    """Structured output forward pass модели RQVAE.

    Attributes:
        x_recon: Реконструированный эмбеддинг ``[B, item_embedding_dim]``.
        indices: Список индексов кодов по уровням (каждый ``[B]``).
        loss: Суммарный loss (recon + vq).
        recon_loss: Reconstruction loss (MSE).
        vq_loss: Суммарный VQ loss по всем уровням.
        codebook_losses: Список codebook losses по уровням (для VectorQuantizer).
        commitment_losses: Список commitment losses по уровням.
        residual: Финальный остаток после всех уровней квантования.
    """

    x_recon: Tensor
    indices: list[Tensor]
    loss: Tensor
    recon_loss: Tensor
    vq_loss: Tensor
    codebook_losses: list[Tensor]
    commitment_losses: list[Tensor]
    residual: Tensor


def torch_kmeans(
    data: Tensor,
    k: int,
    n_iter: int = 10,
    seed: int = 0,
) -> Tensor:
    """K-means кластеризация на PyTorch (GPU-friendly).

    Args:
        data: Данные ``[N, D]``.
        k: Число кластеров.
        n_iter: Число итераций.
        seed: Seed для воспроизводимости.

    Returns:
        Центроиды ``[K, D]``.
    """
    n, d = data.shape
    device = data.device

    # Инициализация: случайные точки из data
    generator = torch.Generator(device=device).manual_seed(seed)
    perm = torch.randperm(n, generator=generator, device=device)[:k]
    centroids = data[perm].clone()

    for _ in range(n_iter):
        # Assign: найти ближайший центроид для каждой точки
        # ||x - c||² = ||x||² + ||c||² - 2⟨x, c⟩
        x_sq = (data ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        c_sq = (centroids ** 2).sum(dim=1)  # [K]
        xc = data @ centroids.T  # [N, K]
        distances = x_sq + c_sq - 2 * xc
        assignments = distances.argmin(dim=1)  # [N]

        # Update: пересчитать центроиды
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=device)

        # scatter_add для суммирования
        for i in range(d):
            new_centroids[:, i].scatter_add_(0, assignments, data[:, i])
        counts.scatter_add_(0, assignments, torch.ones(n, device=device))

        # Избегаем деления на 0
        mask = counts > 0
        new_centroids[mask] /= counts[mask].unsqueeze(1)

        # Для пустых кластеров — оставляем старые центроиды
        new_centroids[~mask] = centroids[~mask]
        centroids = new_centroids

    return centroids


class RQVAE(nn.Module):
    """Residual Quantized VAE для генерации иерархических семантических ID.

    Архитектура:
    - Encoder: MLP, сжимает эмбеддинг [item_embedding_dim] → [codebook_embedding_dim]
    - Residual VQ: несколько уровней квантования с вычитанием остатка
    - Decoder: MLP, восстанавливает [codebook_embedding_dim] → [item_embedding_dim]

    Результат: каждому входному эмбеддингу присваивается semantic ID —
    набор индексов кодов по всем уровням (например, [42, 128, 7]).
    """

    def __init__(self, config: RQVAEConfig) -> None:
        super().__init__()
        self.config = config

        self.item_embedding_dim = config.item_embedding_dim
        self.encoder_hidden_dims = config.encoder_hidden_dims
        self.codebook_embedding_dim = config.codebook_embedding_dim
        self.codebook_quantization_levels = config.codebook_quantization_levels
        self.codebook_size = config.codebook_size

        self.encoder = self._build_encoder(config)
        self.decoder = self._build_decoder(config)

        quantizer_class = EMAVectorQuantizer if config.use_ema_vq else VectorQuantizer
        self.vq_layers = nn.ModuleList([quantizer_class(config) for _ in range(config.codebook_quantization_levels)])

    def _build_encoder(self, config: RQVAEConfig) -> nn.Sequential:
        """Построить encoder: item_embedding_dim → ... → codebook_embedding_dim."""
        layers: list[nn.Module] = []
        dims = [config.item_embedding_dim, *config.encoder_hidden_dims, config.codebook_embedding_dim]
        for i in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.SiLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        return nn.Sequential(*layers)

    def _build_decoder(self, config: RQVAEConfig) -> nn.Sequential:
        """Построить decoder: codebook_embedding_dim → ... → item_embedding_dim."""
        layers: list[nn.Module] = []
        dims = [config.codebook_embedding_dim, *config.encoder_hidden_dims[::-1], config.item_embedding_dim]
        for i in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.SiLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        return nn.Sequential(*layers)

    def encode(self, x: Tensor) -> Tensor:
        """Закодировать входной эмбеддинг в латентное пространство."""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Декодировать латентное представление обратно в пространство эмбеддингов."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> ForwardOutput:
        """Forward pass: encoder → residual VQ → decoder.

        Args:
            x: Входной эмбеддинг ``[B, item_embedding_dim]``.

        Returns:
            ForwardOutput с реконструкцией, индексами и всеми лоссами.
        """
        z = self.encode(x)

        quantized_out = torch.zeros_like(z)
        residual = z

        all_indices: list[Tensor] = []
        vq_loss: Tensor = torch.tensor(0.0, device=x.device)
        codebook_losses: list[Tensor] = []
        commitment_losses: list[Tensor] = []

        for vq_layer in self.vq_layers:
            vq_output = vq_layer(residual)
            residual = residual - vq_output.quantized.detach()
            quantized_out = quantized_out + vq_output.quantized_st
            all_indices.append(vq_output.indices)

            vq_loss = vq_loss + vq_output.loss
            if vq_output.codebook_loss is not None:
                codebook_losses.append(vq_output.codebook_loss)
            commitment_losses.append(vq_output.commitment_loss)

        x_recon = self.decode(quantized_out)
        recon_loss = F.mse_loss(x_recon, x)
        loss = recon_loss + vq_loss

        return ForwardOutput(
            x_recon=x_recon,
            indices=all_indices,
            loss=loss,
            recon_loss=recon_loss,
            vq_loss=vq_loss,
            codebook_losses=codebook_losses,
            commitment_losses=commitment_losses,
            residual=residual,
        )

    @torch.no_grad()
    def encode_to_semantic_ids(self, x: Tensor) -> Tensor:
        """Получить semantic ID для входных эмбеддингов (без градиентов).

        Args:
            x: Входной эмбеддинг ``[B, item_embedding_dim]``.

        Returns:
            Тензор ``[B, L]`` — индексы кодов по каждому уровню.
        """
        z = self.encode(x)
        residual = z
        indices_list = []
        for vq_layer in self.vq_layers:
            vq_output = vq_layer(residual)
            indices_list.append(vq_output.indices)
            residual = residual - vq_output.quantized
        return torch.stack(indices_list, dim=-1)

    @torch.no_grad()
    def decode_from_semantic_ids(self, semantic_ids: Tensor) -> Tensor:
        """Декодировать semantic ID обратно в эмбеддинг.

        Args:
            semantic_ids: Тензор ``[B, L]`` — индексы кодов по уровням.

        Returns:
            Реконструированный эмбеддинг ``[B, item_embedding_dim]``.
        """
        quantized_sum = torch.zeros(semantic_ids.shape[0], self.codebook_embedding_dim, device=semantic_ids.device)
        for level, indices in enumerate(semantic_ids.unbind(dim=-1)):
            codes = self.vq_layers[level].embedding(indices)
            quantized_sum += codes
        return self.decode(quantized_sum)

    @torch.no_grad()
    def kmeans_init(self, data_loader, device: str) -> None:
        """Инициализация кодбуков k-means по первому батчу.

        Использует GPU-friendly torch k-means вместо sklearn.

        Для каждого уровня:
        1. Берём текущий residual
        2. Запускаем k-means
        3. Используем центроиды как начальные веса кодбука
        """
        logger.info("Initializing codebooks with k-means clustering...")

        first_batch = next(iter(data_loader))
        if isinstance(first_batch, (list, tuple)):
            first_batch = first_batch[0]
        first_batch = first_batch.to(device)

        z = self.encode(first_batch)
        residual = z

        for level, vq_layer in enumerate(self.vq_layers):
            centroids = torch_kmeans(residual, self.codebook_size, n_iter=10, seed=level)
            vq_layer.embedding.weight.data = centroids
            logger.info(f"  Level {level}: initialized {self.codebook_size} codes")

            if level < self.codebook_quantization_levels - 1:
                vq_output = vq_layer(residual)
                residual = residual - vq_output.quantized
