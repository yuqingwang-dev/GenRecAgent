"""Small RQ-VAE trained on dense item embeddings."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from genrecagent.semantic_id.quantizer import ResidualQuantizer
from genrecagent.semantic_id.stabilized import PCASkipDecoder, PCASkipEncoder, variance_covariance


@dataclass(frozen=True, slots=True)
class RQVAEConfig:
    input_dim: int
    hidden_dim: int = 256
    latent_dim: int = 64
    num_levels: int = 3
    codebook_size: int = 256
    commitment_weight: float = 0.25
    variant: str = "legacy"
    quantization_weight: float = 1.0
    variance_weight: float = 0.0
    covariance_weight: float = 0.0
    ema_decay: float = 0.99
    dead_code_threshold: float = 0.1

    def __post_init__(self) -> None:
        if (
            min(
                self.input_dim,
                self.hidden_dim,
                self.latent_dim,
                self.num_levels,
                self.codebook_size,
            )
            < 1
        ):
            raise ValueError("RQ-VAE dimensions and codebook settings must be positive")
        if self.variant not in {"legacy", "pca_ema"}:
            raise ValueError("Choose legacy or pca_ema RQ-VAE")
        if self.variant == "pca_ema" and self.latent_dim > self.input_dim:
            raise ValueError("PCA initialization requires latent_dim <= input_dim")
        for value in (
            self.commitment_weight,
            self.quantization_weight,
            self.variance_weight,
            self.covariance_weight,
            self.dead_code_threshold,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Loss weights and dead threshold must be finite/nonnegative")
        if not 0 < self.ema_decay < 1:
            raise ValueError("EMA decay must lie strictly between zero and one")


@dataclass(slots=True)
class RQVAEOutput:
    reconstruction: Tensor
    codes: Tensor
    reconstruction_loss: Tensor
    quantization_loss: Tensor
    total_loss: Tensor
    reconstruction_objective: Tensor
    regularization_loss: Tensor


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
        if config.variant == "pca_ema":
            self.encoder = PCASkipEncoder(config.input_dim, config.hidden_dim, config.latent_dim)
            self.decoder = PCASkipDecoder(config.input_dim, config.hidden_dim, config.latent_dim)
            self.quantizer = ResidualQuantizer(
                config.latent_dim,
                config.num_levels,
                config.codebook_size,
                config.commitment_weight,
                update_method="ema",
                ema_decay=config.ema_decay,
                dead_code_threshold=config.dead_code_threshold,
            )
            self.register_buffer("pca_initialized", torch.tensor(False))

    @torch.no_grad()
    def initialize_pca(self, train_vectors, *, seed: int) -> None:
        if self.config.variant != "pca_ema":
            raise ValueError("PCA initialization belongs to the explicit stabilized variant")
        from genrecagent.semantic_id.projection import PCAWhitening, PCAWhiteningConfig

        projector = PCAWhitening(
            PCAWhiteningConfig(self.config.input_dim, self.config.latent_dim, seed)
        ).fit(train_vectors)
        device = self.encoder.projection.weight.device
        mean = torch.as_tensor(projector.mean, device=device)
        components = torch.as_tensor(projector.components, device=device)
        scales = torch.as_tensor(projector.scales, device=device)
        encoder_weight = components / scales.unsqueeze(1)
        self.encoder.projection.weight.copy_(encoder_weight)
        self.encoder.projection.bias.copy_(-encoder_weight @ mean)
        self.decoder.projection.weight.copy_(components.T * scales.unsqueeze(0))
        self.decoder.projection.bias.copy_(mean)
        self.pca_initialized.fill_(True)

    def _regularization(self, latents: Tensor) -> Tensor:
        if self.config.variance_weight == self.config.covariance_weight == 0:
            return latents.new_zeros(())
        variance, covariance = variance_covariance(latents)
        return self.config.variance_weight * variance + self.config.covariance_weight * covariance

    def warmup_loss(self, vectors: Tensor) -> Tensor:
        if self.config.variant != "pca_ema" or not self.pca_initialized:
            raise RuntimeError("Stabilized warmup requires train-only PCA initialization")
        latents = self.encoder(vectors)
        reconstruction = F.normalize(self.decoder(latents), dim=1)
        return (1 - F.cosine_similarity(reconstruction, vectors, dim=1)).mean() + (
            self._regularization(latents)
        )

    def forward(self, vectors: Tensor) -> RQVAEOutput:
        if vectors.ndim != 2 or vectors.shape[1] != self.config.input_dim:
            raise ValueError("Expected [batch, input_dim] vectors")
        if self.config.variant == "pca_ema" and not self.pca_initialized:
            raise RuntimeError("Initialize stabilized encoder from training vectors first")
        latents = self.encoder(vectors)
        quantized, codes, quantization_loss = self.quantizer(latents)
        reconstruction = F.normalize(self.decoder(quantized), dim=1)
        reconstruction_loss = F.mse_loss(reconstruction, vectors)
        objective = (
            reconstruction_loss
            if self.config.variant == "legacy"
            else (1 - F.cosine_similarity(reconstruction, vectors, dim=1)).mean()
        )
        regularization = self._regularization(latents)
        return RQVAEOutput(
            reconstruction=reconstruction,
            codes=codes,
            reconstruction_loss=reconstruction_loss,
            quantization_loss=quantization_loss,
            total_loss=objective
            + self.config.quantization_weight * quantization_loss
            + regularization,
            reconstruction_objective=objective,
            regularization_loss=regularization,
        )

    @torch.no_grad()
    def encode(self, vectors: Tensor) -> Tensor:
        if self.config.variant == "pca_ema" and not self.pca_initialized:
            raise RuntimeError("Initialize stabilized encoder first")
        return self.quantizer(self.encoder(vectors), update_stats=False)[1]

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
