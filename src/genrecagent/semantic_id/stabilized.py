"""Trainable PCA-warmstarted bottleneck and variance/covariance regularization."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PCASkipEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, latent_dim)
        self.residual = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.radius = math.sqrt(latent_dim)

    def forward(self, vectors: Tensor) -> Tensor:
        projected = self.projection(vectors)
        return F.normalize(projected + self.residual(projected), dim=1, eps=1e-6) * self.radius


class PCASkipDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.projection = nn.Linear(latent_dim, input_dim)
        self.residual = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, input_dim)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, latents: Tensor) -> Tensor:
        return self.projection(latents) + self.residual(latents)


def variance_covariance(latents: Tensor, *, floor: float = 0.8) -> tuple[Tensor, Tensor]:
    centered = latents - latents.mean(0)
    standard_deviation = (centered.square().mean(0) + 1e-4).sqrt()
    variance = F.relu(floor - standard_deviation).mean()
    covariance = centered.T @ centered / max(len(latents) - 1, 1)
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    decorrelation = off_diagonal.square().sum() / latents.shape[1]
    return variance, decorrelation
