"""Trainable residual vector quantization for RQ-VAE."""

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
    ) -> None:
        super().__init__()
        if min(latent_dim, num_levels, codebook_size) < 1:
            raise ValueError("latent_dim, num_levels, and codebook_size must be positive")
        if commitment_weight < 0:
            raise ValueError("commitment_weight cannot be negative")
        self.latent_dim = latent_dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.codebooks = nn.ModuleList(
            nn.Embedding(codebook_size, latent_dim) for _ in range(num_levels)
        )
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    @torch.no_grad()
    def initialize_from_latents(self, latents: Tensor, *, seed: int) -> None:
        """Initialize each residual codebook with train-only k-means++ seeds."""

        if latents.ndim != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError("Expected [sample, latent_dim] train latents")
        if len(latents) < self.codebook_size:
            raise ValueError("Initialization needs at least codebook_size training samples")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        residual = latents.detach().to(device="cpu", dtype=torch.float32)
        for codebook in self.codebooks:
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
            codebook.weight.copy_(centers.to(codebook.weight.device))
            distances = (
                residual.square().sum(dim=1, keepdim=True)
                + centers.square().sum(dim=1).unsqueeze(0)
                - 2 * residual @ centers.T
            )
            residual = residual - centers[distances.argmin(dim=1)]

    def forward(self, latents: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if latents.ndim != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError("Expected [batch, latent_dim] latents")
        residual = latents
        quantized_sum = torch.zeros_like(latents)
        losses: list[Tensor] = []
        indices: list[Tensor] = []
        for codebook in self.codebooks:
            weights = codebook.weight
            distances = (
                residual.square().sum(dim=1, keepdim=True)
                + weights.square().sum(dim=1).unsqueeze(0)
                - 2 * residual @ weights.T
            )
            selected = distances.argmin(dim=1)
            quantized = codebook(selected)
            losses.append(
                F.mse_loss(quantized, residual.detach())
                + self.commitment_weight * F.mse_loss(residual, quantized.detach())
            )
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized.detach()
            indices.append(selected)
        straight_through = latents + (quantized_sum - latents).detach()
        return straight_through, torch.stack(indices, dim=1), torch.stack(losses).sum()

    def embed_codes(self, codes: Tensor) -> Tensor:
        if codes.ndim != 2 or codes.shape[1] != self.num_levels:
            raise ValueError("Expected [batch, num_levels] codes")
        if codes.numel() and (codes.min() < 0 or codes.max() >= self.codebook_size):
            raise ValueError("Code index outside the codebook")
        return sum(
            codebook(codes[:, level])
            for level, codebook in enumerate(self.codebooks)
        )
