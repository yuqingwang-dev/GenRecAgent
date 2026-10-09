"""Trainable residual vector quantization for RQ-VAE."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualQuantizer(nn.Module):
    """Nearest-code residual quantization with straight-through gradients."""

    def __init__(
        self,
        latent_dim: int,
        num_levels: int,
        codebook_size: int,
        commitment_weight: float = 0.25,
        *,
        update_method: str = "sgd",
        ema_decay: float = 0.99,
        dead_code_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        if min(latent_dim, num_levels, codebook_size) < 1:
            raise ValueError("latent_dim, num_levels, and codebook_size must be positive")
        if not math.isfinite(commitment_weight) or commitment_weight < 0:
            raise ValueError("commitment_weight cannot be negative")
        if update_method not in {"sgd", "ema"} or not 0 < ema_decay < 1:
            raise ValueError("Choose sgd/ema and a decay strictly between zero and one")
        if not math.isfinite(dead_code_threshold) or dead_code_threshold < 0:
            raise ValueError("Dead-code threshold must be finite and nonnegative")
        self.latent_dim = latent_dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.update_method, self.ema_decay = update_method, ema_decay
        self.dead_code_threshold = dead_code_threshold
        self.codebooks = nn.ModuleList(
            nn.Embedding(codebook_size, latent_dim) for _ in range(num_levels)
        )
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)
            if update_method == "ema":
                codebook.weight.requires_grad_(False)
        if update_method == "ema":
            self.register_buffer("ema_counts", torch.ones(num_levels, codebook_size))
            self.register_buffer(
                "ema_sums", torch.stack([book.weight.detach().clone() for book in self.codebooks])
            )
            self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))
            self.register_buffer("dead_resets", torch.zeros(num_levels, dtype=torch.long))

    @torch.no_grad()
    def initialize_from_latents(
        self, latents: Tensor, *, seed: int, lloyd_iterations: int = 0
    ) -> None:
        """Initialize each residual codebook with train-only k-means++ seeds."""

        if latents.ndim != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError("Expected [sample, latent_dim] train latents")
        if len(latents) < self.codebook_size:
            raise ValueError("Initialization needs at least codebook_size training samples")
        if not torch.isfinite(latents).all() or lloyd_iterations < 0:
            raise ValueError("Initialization needs finite latents and nonnegative iterations")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        residual = latents.detach().to(device="cpu", dtype=torch.float32)
        for level, codebook in enumerate(self.codebooks):
            centers = torch.empty((self.codebook_size, self.latent_dim))
            minimum_distance = torch.full((len(residual),), float("inf"))
            for code in range(self.codebook_size):
                if code == 0 or float(minimum_distance.sum()) == 0:
                    index = int(torch.randint(len(residual), (1,), generator=generator))
                else:
                    index = int(torch.multinomial(minimum_distance, 1, generator=generator))
                centers[code] = residual[index]
                distances = (residual - centers[code]).square().sum(dim=1)
                minimum_distance = torch.minimum(minimum_distance, distances)
            for _ in range(lloyd_iterations):
                distance = (
                    residual.square().sum(1, keepdim=True)
                    + centers.square().sum(1).unsqueeze(0)
                    - 2 * residual @ centers.T
                )
                labels = distance.argmin(1)
                counts = torch.bincount(labels, minlength=self.codebook_size)
                sums = torch.zeros_like(centers).index_add_(0, labels, residual)
                active = counts > 0
                centers[active] = sums[active] / counts[active].unsqueeze(1)
                if (~active).any():
                    chosen = torch.randint(
                        len(residual), (int((~active).sum()),), generator=generator
                    )
                    centers[~active] = residual[chosen]
            codebook.weight.copy_(centers.to(codebook.weight.device))
            if self.update_method == "ema":
                self.ema_counts[level].fill_(1)
                self.ema_sums[level].copy_(codebook.weight)
            distances = (
                residual.square().sum(dim=1, keepdim=True)
                + centers.square().sum(dim=1).unsqueeze(0)
                - 2 * residual @ centers.T
            )
            residual = residual - centers[distances.argmin(dim=1)]

    @torch.no_grad()
    def _ema_update(self, level: int, residual: Tensor, labels: Tensor) -> None:
        values = residual.detach()
        counts = torch.bincount(labels, minlength=self.codebook_size).to(values.dtype)
        sums = torch.zeros_like(self.ema_sums[level]).index_add_(0, labels, values)
        self.ema_counts[level].mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)
        self.ema_sums[level].mul_(self.ema_decay).add_(sums, alpha=1 - self.ema_decay)
        dead = self.ema_counts[level] < self.dead_code_threshold
        if dead.any():
            chosen = torch.randint(len(values), (int(dead.sum()),), device=values.device)
            self.ema_counts[level][dead] = 1
            self.ema_sums[level][dead] = values[chosen]
            self.dead_resets[level].add_(dead.sum())
        self.codebooks[level].weight.copy_(
            self.ema_sums[level] / self.ema_counts[level].clamp_min(1e-6).unsqueeze(1)
        )

    def forward(
        self, latents: Tensor, *, update_stats: bool = True
    ) -> tuple[Tensor, Tensor, Tensor]:
        if latents.ndim != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError("Expected [batch, latent_dim] latents")
        residual = latents
        quantized_sum = torch.zeros_like(latents)
        losses: list[Tensor] = []
        indices: list[Tensor] = []
        for level, codebook in enumerate(self.codebooks):
            weights = codebook.weight
            distances = (
                residual.square().sum(dim=1, keepdim=True)
                + weights.square().sum(dim=1).unsqueeze(0)
                - 2 * residual @ weights.T
            )
            selected = distances.argmin(dim=1)
            quantized = codebook(selected)
            commitment = self.commitment_weight * F.mse_loss(residual, quantized.detach())
            losses.append(
                commitment
                if self.update_method == "ema"
                else (F.mse_loss(quantized, residual.detach()) + commitment)
            )
            if self.update_method == "ema" and self.training and update_stats:
                self._ema_update(level, residual, selected)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized.detach()
            indices.append(selected)
        if self.update_method == "ema" and self.training and update_stats:
            self.ema_updates.add_(1)
        straight_through = latents + (quantized_sum - latents).detach()
        return straight_through, torch.stack(indices, dim=1), torch.stack(losses).sum()

    def embed_codes(self, codes: Tensor) -> Tensor:
        if codes.ndim != 2 or codes.shape[1] != self.num_levels:
            raise ValueError("Expected [batch, num_levels] codes")
        if codes.numel() and (codes.min() < 0 or codes.max() >= self.codebook_size):
            raise ValueError("Code index outside the codebook")
        return sum(codebook(codes[:, level]) for level, codebook in enumerate(self.codebooks))
