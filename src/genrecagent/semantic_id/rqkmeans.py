"""Residual MiniBatch K-Means baseline for item Semantic IDs.

This is a separately named clustering baseline, not an RQ-VAE model.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass(frozen=True, slots=True)
class RQKMeansConfig:
    input_dim: int
    num_levels: int = 3
    codebook_size: int = 256
    batch_size: int = 4096
    max_iter: int = 30
    seed: int = 42

    def __post_init__(self) -> None:
        if min(
            self.input_dim, self.num_levels, self.codebook_size,
            self.batch_size, self.max_iter,
        ) < 1:
            raise ValueError("RQ-KMeans dimensions and iteration settings must be positive")


class ResidualKMeans:
    """Fit codebooks level by level on training vectors only."""

    def __init__(self, config: RQKMeansConfig) -> None:
        self.config = config
        self.codebooks: list[np.ndarray] = []
        self.fit_log: list[dict[str, float | int]] = []

    def fit(self, vectors: np.ndarray) -> "ResidualKMeans":
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.input_dim:
            raise ValueError("Expected [items, input_dim] training vectors")
        if len(values) < self.config.codebook_size or not np.isfinite(values).all():
            raise ValueError("Need at least codebook_size finite training vectors")
        residual = values.copy()
        self.codebooks = []
        self.fit_log = []
        for level in range(self.config.num_levels):
            started = perf_counter()
            clusterer = MiniBatchKMeans(
                n_clusters=self.config.codebook_size,
                init="k-means++",
                n_init=3,
                batch_size=self.config.batch_size,
                max_iter=self.config.max_iter,
                random_state=self.config.seed + level,
                reassignment_ratio=0.01,
                compute_labels=False,
            )
            clusterer.fit(residual)
            centers = np.asarray(clusterer.cluster_centers_, dtype=np.float32)
            labels = clusterer.predict(residual)
            residual -= centers[labels]
            self.codebooks.append(centers)
            record: dict[str, float | int] = {
                "level": level,
                "iterations": int(clusterer.n_iter_),
                "steps": int(clusterer.n_steps_),
                "train_residual_mse": float(np.mean(residual**2)),
                "train_used_codes": int(np.unique(labels).size),
                "elapsed_seconds": round(perf_counter() - started, 3),
            }
            self.fit_log.append(record)
            print(
                f"level={level} train_residual_mse={record['train_residual_mse']:.6f} "
                f"used_codes={record['train_used_codes']}/{self.config.codebook_size}",
                flush=True,
            )
        return self

    def encode(
        self, vectors: np.ndarray, *, levels: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.codebooks:
            raise RuntimeError("Fit or load codebooks before encoding")
        count = len(self.codebooks) if levels is None else levels
        if not 1 <= count <= len(self.codebooks):
            raise ValueError("levels must be in the fitted range")
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.input_dim:
            raise ValueError("Expected [items, input_dim] vectors")
        residual = values.copy()
        codes = np.empty((len(values), count), dtype=np.int32)
        for level, centers in enumerate(self.codebooks[:count]):
            center_norms = np.einsum("ij,ij->i", centers, centers)
            for start in range(0, len(values), self.config.batch_size):
                stop = min(start + self.config.batch_size, len(values))
                batch = residual[start:stop]
                distances = (
                    np.einsum("ij,ij->i", batch, batch)[:, None]
                    + center_norms[None, :]
                    - 2.0 * batch @ centers.T
                )
                selected = distances.argmin(axis=1)
                codes[start:stop, level] = selected
                residual[start:stop] -= centers[selected]
        return codes, residual

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        if not self.codebooks:
            raise RuntimeError("Fit or load codebooks before reconstruction")
        indices = np.asarray(codes)
        if indices.ndim != 2 or indices.shape[1] != len(self.codebooks):
            raise ValueError("Expected [items, num_levels] codes")
        if indices.size and (indices.min() < 0 or indices.max() >= self.config.codebook_size):
            raise ValueError("Code index outside the codebook")
        result = np.zeros((len(indices), self.config.input_dim), dtype=np.float32)
        for level, centers in enumerate(self.codebooks):
            result += centers[indices[:, level]]
        return result

    def save(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(path)
        if len(self.codebooks) != self.config.num_levels:
            raise RuntimeError("All residual levels must be fitted before saving")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            codebooks=np.stack(self.codebooks),
            **{key: np.asarray(value) for key, value in asdict(self.config).items()},
        )

    @classmethod
    def load(cls, path: Path) -> "ResidualKMeans":
        with np.load(path, allow_pickle=False) as archive:
            config = RQKMeansConfig(**{
                key: int(archive[key]) for key in RQKMeansConfig.__dataclass_fields__
            })
            centers = np.asarray(archive["codebooks"], dtype=np.float32)
        if centers.shape != (
            config.num_levels, config.codebook_size, config.input_dim
        ):
            raise ValueError("Saved codebooks do not match their configuration")
        model = cls(config)
        model.codebooks = [centers[level] for level in range(config.num_levels)]
        return model
