"""Train-only PCA whitening before the residual K-Means baseline."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True, slots=True)
class PCAWhiteningConfig:
    input_dim: int
    output_dim: int
    seed: int = 42

    def __post_init__(self) -> None:
        if self.input_dim < 1 or not 1 <= self.output_dim <= self.input_dim:
            raise ValueError("PCA dimensions must be positive and output_dim <= input_dim")


class PCAWhitening:
    """Persist PCA parameters as safe NumPy arrays, not a pickle."""

    def __init__(self, config: PCAWhiteningConfig) -> None:
        self.config = config
        self.mean: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.scales: np.ndarray | None = None

    def fit(self, vectors: np.ndarray) -> "PCAWhitening":
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.input_dim:
            raise ValueError("Expected [train_items, input_dim] vectors")
        if len(values) < self.config.output_dim or not np.isfinite(values).all():
            raise ValueError("PCA needs enough finite training vectors")
        estimator = PCA(
            n_components=self.config.output_dim,
            whiten=True,
            svd_solver="randomized" if self.config.output_dim < self.config.input_dim else "full",
            random_state=self.config.seed,
        ).fit(values)
        self.mean = np.asarray(estimator.mean_, dtype=np.float32)
        self.components = np.asarray(estimator.components_, dtype=np.float32)
        self.scales = np.sqrt(np.asarray(estimator.explained_variance_, dtype=np.float32))
        if np.any(self.scales <= 1e-10):
            raise ValueError("PCA components include near-zero variance directions")
        return self

    def _fitted(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.mean is None or self.components is None or self.scales is None:
            raise RuntimeError("Fit or load PCA before transforming vectors")
        return self.mean, self.components, self.scales

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        mean, components, scales = self._fitted()
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.input_dim:
            raise ValueError("Expected [items, input_dim] vectors")
        return np.asarray(((values - mean) @ components.T) / scales, dtype=np.float32)

    def inverse_transform(self, vectors: np.ndarray) -> np.ndarray:
        mean, components, scales = self._fitted()
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.output_dim:
            raise ValueError("Expected [items, output_dim] vectors")
        return np.asarray((values * scales) @ components + mean, dtype=np.float32)

    def save(self, path: Path) -> None:
        mean, components, scales = self._fitted()
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, mean=mean, components=components, scales=scales,
            **{key: np.asarray(value) for key, value in asdict(self.config).items()},
        )

    @classmethod
    def load(cls, path: Path) -> "PCAWhitening":
        with np.load(path, allow_pickle=False) as archive:
            config = PCAWhiteningConfig(**{
                key: int(archive[key]) for key in PCAWhiteningConfig.__dataclass_fields__
            })
            mean = np.asarray(archive["mean"], dtype=np.float32)
            components = np.asarray(archive["components"], dtype=np.float32)
            scales = np.asarray(archive["scales"], dtype=np.float32)
        if mean.shape != (config.input_dim,) or components.shape != (
            config.output_dim, config.input_dim
        ) or scales.shape != (config.output_dim,):
            raise ValueError("Saved PCA arrays do not match their configuration")
        result = cls(config)
        result.mean, result.components, result.scales = mean, components, scales
        return result
