"""Reproducible RQ-VAE training and full-catalog Semantic ID encoding."""

import hashlib
import math
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import torch

from genrecagent.semantic_id.codebook import SemanticCodebook
from genrecagent.semantic_id.rqvae import RQVAE, RQVAEConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 1e-3
    validation_fraction: float = 0.1
    seed: int = 42
    device: str = "auto"
    init_samples: int = 4096

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie between zero and one")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.init_samples < 1:
            raise ValueError("init_samples must be positive")


@dataclass(slots=True)
class TrainingResult:
    model: RQVAE
    best_epoch: int
    best_validation_reconstruction_mse: float
    train_indices: np.ndarray
    validation_indices: np.ndarray
    epoch_log: list[dict[str, float | int]]
    device: str


def split_item_indices(
    item_ids: list[str], *, seed: int, validation_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    if len(item_ids) < 2 or len(set(item_ids)) != len(item_ids):
        raise ValueError("At least two unique item IDs are required")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie between zero and one")
    ordered = sorted(
        range(len(item_ids)),
        key=lambda index: (
            hashlib.sha256(f"{seed}:{item_ids[index]}".encode()).digest(), item_ids[index]
        ),
    )
    count = min(len(item_ids) - 1, max(1, round(len(item_ids) * validation_fraction)))
    return np.asarray(ordered[count:], dtype=np.int64), np.asarray(ordered[:count], dtype=np.int64)


def _device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def _batches(indices: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def fit_rqvae(
    vectors: np.ndarray,
    item_ids: list[str],
    model_config: RQVAEConfig,
    train_config: TrainingConfig,
) -> TrainingResult:
    if vectors.ndim != 2 or vectors.shape != (len(item_ids), model_config.input_dim):
        raise ValueError("Item IDs and embedding dimensions must match the RQ-VAE input")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be finite")
    train_indices, validation_indices = split_item_indices(
        item_ids,
        seed=train_config.seed,
        validation_fraction=train_config.validation_fraction,
    )
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = _device(train_config.device)
    model = RQVAE(model_config).to(device)
    rng = np.random.default_rng(train_config.seed)
    init_indices = rng.choice(
        train_indices,
        size=min(train_config.init_samples, len(train_indices)),
        replace=False,
    )
    with torch.no_grad():
        init_batch = torch.as_tensor(
            np.asarray(vectors[init_indices], dtype=np.float32), device=device
        )
        model.quantizer.initialize_from_latents(
            model.encoder(init_batch), seed=train_config.seed
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    best_loss = math.inf
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    epoch_log: list[dict[str, float | int]] = []
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        running_train = 0.0
        shuffled = rng.permutation(train_indices)
        for batch_indices in _batches(shuffled, train_config.batch_size):
            batch = torch.as_tensor(
                np.asarray(vectors[batch_indices], dtype=np.float32), device=device
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            output.total_loss.backward()
            optimizer.step()
            running_train += float(output.total_loss.detach()) * len(batch_indices)
        model.eval()
        validation_reconstruction = 0.0
        validation_total = 0.0
        with torch.no_grad():
            for batch_indices in _batches(validation_indices, train_config.batch_size):
                batch = torch.as_tensor(
                    np.asarray(vectors[batch_indices], dtype=np.float32), device=device
                )
                output = model(batch)
                validation_reconstruction += float(output.reconstruction_loss) * len(batch_indices)
                validation_total += float(output.total_loss) * len(batch_indices)
        validation_reconstruction /= len(validation_indices)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_total_loss": running_train / len(train_indices),
            "validation_reconstruction_mse": validation_reconstruction,
            "validation_total_loss": validation_total / len(validation_indices),
        }
        epoch_log.append(record)
        print(
            f"epoch={epoch} train_total={record['train_total_loss']:.6f} "
            f"val_reconstruction_mse={validation_reconstruction:.6f}",
            flush=True,
        )
        if validation_reconstruction < best_loss:
            best_loss = validation_reconstruction
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_validation_reconstruction_mse=best_loss,
        train_indices=train_indices,
        validation_indices=validation_indices,
        epoch_log=epoch_log,
        device=device,
    )


@torch.no_grad()
def encode_catalog(
    model: RQVAE,
    vectors: np.ndarray,
    item_ids: list[str],
    *,
    batch_size: int,
    device: str,
) -> tuple[SemanticCodebook, dict[str, object]]:
    if batch_size < 1 or not item_ids:
        raise ValueError("batch_size and catalog size must be positive")
    if vectors.shape != (len(item_ids), model.config.input_dim):
        raise ValueError("Catalog vectors and item IDs must align")
    model = model.to(device).eval()
    mapping = SemanticCodebook()
    counts = np.zeros((model.config.num_levels, model.config.codebook_size), dtype=np.int64)
    squared_error = 0.0
    for start in range(0, len(item_ids), batch_size):
        end = min(start + batch_size, len(item_ids))
        batch = torch.as_tensor(np.asarray(vectors[start:end], dtype=np.float32), device=device)
        output = model(batch)
        codes = output.codes.cpu().numpy()
        squared_error += float(output.reconstruction_loss) * (end - start)
        for item_id, item_codes in zip(item_ids[start:end], codes, strict=True):
            mapping.add(
                item_id,
                tuple(f"<s{level}:{int(code)}>" for level, code in enumerate(item_codes)),
            )
        for level in range(model.config.num_levels):
            counts[level] += np.bincount(codes[:, level], minlength=model.config.codebook_size)
    usage = []
    for level_counts in counts:
        probabilities = level_counts[level_counts > 0] / max(1, level_counts.sum())
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        usage.append(
            {
                "used_codes": int(np.count_nonzero(level_counts)),
                "codebook_size": model.config.codebook_size,
                "utilization": float(np.count_nonzero(level_counts) / model.config.codebook_size),
                "perplexity": float(math.exp(entropy)),
            }
        )
    metrics: dict[str, object] = {
        "catalog_reconstruction_mse": squared_error / len(item_ids),
        "codebook_usage_by_level": usage,
        "collisions": mapping.collision_stats(),
    }
    mapping.metadata = {
        "model_type": "trained_rqvae",
        "config": asdict(model.config),
        "collision_policy": "lookup returns every catalog item with the generated code",
    }
    return mapping, metrics
