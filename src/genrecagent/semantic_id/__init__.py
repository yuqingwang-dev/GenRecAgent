"""Semantic-ID extension points."""

from genrecagent.semantic_id.codebook import SemanticCodebook
from genrecagent.semantic_id.projection import PCAWhitening, PCAWhiteningConfig
from genrecagent.semantic_id.quantizer import ResidualQuantizer
from genrecagent.semantic_id.rqkmeans import ResidualKMeans, RQKMeansConfig
from genrecagent.semantic_id.rqvae import RQVAE, RQVAEConfig

__all__ = [
    "PCAWhitening", "PCAWhiteningConfig", "RQVAE", "RQVAEConfig",
    "RQKMeansConfig", "ResidualKMeans",
    "ResidualQuantizer", "SemanticCodebook",
]
