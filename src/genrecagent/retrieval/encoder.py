"""Sentence Transformer item encoder with an explicit local cache and download gate."""

from pathlib import Path
from typing import Protocol

import numpy as np


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Expected a two-dimensional embedding matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    values = np.asarray(vector, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional vector")
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 0 and np.isfinite(norm) else None


class TextEncoder(Protocol):
    model_name: str
    device: str

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode metadata texts to normalized float32 vectors."""


class SentenceItemEncoder:
    """BGE-compatible SentenceTransformer wrapper; model loading is opt-in."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path,
        device: str = "auto",
        allow_download: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers in genrecagent first") from exc
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=str(cache_dir),
            local_files_only=not allow_download,
            trust_remote_code=False,
        )

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        )
        return normalize_rows(np.asarray(vectors, dtype=np.float32))
