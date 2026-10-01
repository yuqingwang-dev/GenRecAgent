"""Semantic-ID extension points."""

from genrecagent.semantic_id.codebook import SemanticCodebook
from genrecagent.semantic_id.quantizer import ResidualQuantizer
from genrecagent.semantic_id.rqvae import RQVAE, RQVAEConfig

__all__ = ["RQVAE", "RQVAEConfig", "ResidualQuantizer", "SemanticCodebook"]
