"""Quantization extension point for RQ-KMeans or RQ-VAE."""


class ResidualQuantizer:
    """Placeholder with an explicit API for future learned quantization."""

    def fit(self, vectors: list[list[float]]) -> None:
        del vectors
        raise NotImplementedError("RQ-KMeans/RQ-VAE is planned for Phase 2.")

    def encode(self, vector: list[float]) -> tuple[str, ...]:
        del vector
        raise NotImplementedError("Semantic-ID encoding is planned for Phase 2.")

