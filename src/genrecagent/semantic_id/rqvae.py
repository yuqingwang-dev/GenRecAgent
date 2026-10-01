"""Small RQ-VAE trained on dense item embeddings."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from genrecagent.semantic_id.quantizer import ResidualQuantizer


@dataclass(frozen=True, slots=True)
class RQVAEConfig:
    input_dim: int
    hidden_dim: int = 256
    latent_dim: int = 64
    num_levels: int = 3
    codebook_size: int = 256
    commitment_weight: float = 0.25

    def __post_init__(self) -> None:
        if min(
            self.input_dim, self.hidden_dim, self.latent_dim,
            self.num_levels, self.codebook_size,
        ) < 1:
            raise ValueError("RQ-VAE dimensions and codebook settings must be positive")


@dataclass(slots=True)
class RQVAEOutput:
    reconstruction: Tensor
    codes: Tensor
    reconstruction_loss: Tensor
    quantization_loss: Tensor
    total_loss: Tensor


class RQVAE(nn.Module):
    def __init__(self, config: RQVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.quantizer = ResidualQuantizer(
            config.latent_dim,
            config.num_levels,
            config.codebook_size,
            config.commitment_weight,
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def forward(self, vectors: Tensor) -> RQVAEOutput:
        if vectors.ndim != 2 or vectors.shape[1] != self.config.input_dim:
            raise ValueError("Expected [batch, input_dim] vectors")
        latents = self.encoder(vectors)
        quantized, codes, quantization_loss = self.quantizer(latents)
        reconstruction = F.normalize(self.decoder(quantized), dim=1)
        reconstruction_loss = F.mse_loss(reconstruction, vectors)
        return RQVAEOutput(
            reconstruction=reconstruction,
            codes=codes,
            reconstruction_loss=reconstruction_loss,
            quantization_loss=quantization_loss,
            total_loss=reconstruction_loss + quantization_loss,
        )

    @torch.no_grad()
    def encode(self, vectors: Tensor) -> Tensor:
        return self.quantizer(self.encoder(vectors))[1]

    @torch.no_grad()
    def decode(self, codes: Tensor) -> Tensor:
        return F.normalize(self.decoder(self.quantizer.embed_codes(codes)), dim=1)

    def save(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, *, map_location: str = "cpu") -> "RQVAE":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(RQVAEConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model
