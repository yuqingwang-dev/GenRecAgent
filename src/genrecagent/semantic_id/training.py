"""Reproducible RQ-VAE training and full-catalog Semantic ID encoding."""

import hashlib
import math
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from genrecagent.semantic_id.codebook import SemanticCodebook
from genrecagent.semantic_id.quality import checkpoint_score, code_statistics, quality_gate
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
    warmup_epochs: int = 0
    pca_fit_samples: int = 32768
    lloyd_iterations: int = 0
    gradient_clip: float = 0.0
    min_utilization: float = 0.0
    min_perplexity_fraction: float = 0.0
    min_unique_ratio: float = 0.0
    max_group_fraction: float = 1.0
    max_reconstruction_mse: float | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie between zero and one")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.init_samples < 1:
            raise ValueError("init_samples must be positive")
        if self.warmup_epochs < 0 or self.pca_fit_samples < 1 or self.lloyd_iterations < 0:
            raise ValueError("Invalid warmup, PCA sample count or Lloyd iterations")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip < 0:
            raise ValueError("Gradient clip must be finite/nonnegative")
        for value in (
            self.min_utilization,
            self.min_perplexity_fraction,
            self.min_unique_ratio,
            self.max_group_fraction,
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Quality fractions must be finite and in [0,1]")
        if self.max_reconstruction_mse is not None and (
            not math.isfinite(self.max_reconstruction_mse) or self.max_reconstruction_mse <= 0
        ):
            raise ValueError("Reconstruction ceiling must be finite and positive")


@dataclass(slots=True)
class TrainingResult:
    model: RQVAE
    best_epoch: int
    best_validation_reconstruction_mse: float
    train_indices: np.ndarray
    validation_indices: np.ndarray
    epoch_log: list[dict]
    device: str
    warmup_log: list[dict] = field(default_factory=list)
    selected_validation_quality: dict = field(default_factory=dict)
    initialization: dict = field(default_factory=dict)


def _gate(stats: dict, config: TrainingConfig) -> dict:
    return quality_gate(
        stats,
        **{
            key: getattr(config, key)
            for key in (
                "min_utilization",
                "min_perplexity_fraction",
                "min_unique_ratio",
                "max_group_fraction",
                "max_reconstruction_mse",
            )
        },
    )


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
            hashlib.sha256(f"{seed}:{item_ids[index]}".encode()).digest(),
            item_ids[index],
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
        yield indices[start : start + batch_size]


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
    if len(train_indices) < model_config.codebook_size:
        raise ValueError("Training split cannot initialize the declared codebook")
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=train_config.learning_rate
    )
    warmup_log, initialization = [], {"scope": "training items only"}
    if model_config.variant == "pca_ema":
        pca_indices = rng.choice(
            train_indices, size=min(train_config.pca_fit_samples, len(train_indices)), replace=False
        )
        model.initialize_pca(
            np.asarray(vectors[pca_indices], dtype=np.float32), seed=train_config.seed
        )
        initialization.update(
            pca_item_count=len(pca_indices),
            pca_item_ids_sha256=hashlib.sha256(
                "\n".join(item_ids[i] for i in pca_indices).encode()
            ).hexdigest(),
        )
        for epoch in range(1, train_config.warmup_epochs + 1):
            model.train()
            total = 0.0
            for indices in _batches(rng.permutation(train_indices), train_config.batch_size):
                batch = torch.as_tensor(
                    np.asarray(vectors[indices], dtype=np.float32), device=device
                )
                optimizer.zero_grad(set_to_none=True)
                loss = model.warmup_loss(batch)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite autoencoder warmup loss")
                loss.backward()
                if train_config.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), train_config.gradient_clip, error_if_nonfinite=True
                    )
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            record = {"epoch": epoch, "train_warmup_objective": total / len(train_indices)}
            warmup_log.append(record)
            print(f"warmup={epoch} objective={record['train_warmup_objective']:.6f}", flush=True)
    elif train_config.warmup_epochs:
        raise ValueError("Warmup is explicit to the stabilized variant, not the legacy baseline")
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
            model.encoder(init_batch),
            seed=train_config.seed,
            lloyd_iterations=train_config.lloyd_iterations,
        )
    initialization.update(
        latent_init_item_count=len(init_indices), lloyd_iterations=train_config.lloyd_iterations
    )
    best_loss = math.inf
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    epoch_log: list[dict] = []
    best_score, selected_quality = None, {}
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
            if not torch.isfinite(output.total_loss):
                raise ValueError("Nonfinite RQ-VAE training objective")
            output.total_loss.backward()
            if train_config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.gradient_clip, error_if_nonfinite=True
                )
            optimizer.step()
            running_train += float(output.total_loss.detach()) * len(batch_indices)
        model.eval()
        validation_reconstruction = 0.0
        validation_total = 0.0
        validation_codes = []
        with torch.no_grad():
            for batch_indices in _batches(validation_indices, train_config.batch_size):
                batch = torch.as_tensor(
                    np.asarray(vectors[batch_indices], dtype=np.float32), device=device
                )
                output = model(batch)
                if not torch.isfinite(output.total_loss):
                    raise ValueError("Nonfinite RQ-VAE validation objective")
                validation_reconstruction += float(output.reconstruction_loss) * len(batch_indices)
                validation_total += float(output.total_loss) * len(batch_indices)
                validation_codes.append(output.codes.cpu().numpy())
        validation_reconstruction /= len(validation_indices)
        statistics = code_statistics(np.concatenate(validation_codes), model_config.codebook_size)
        statistics["reconstruction_mse"] = validation_reconstruction
        gate = _gate(statistics, train_config)
        record: dict = {
            "epoch": epoch,
            "train_total_loss": running_train / len(train_indices),
            "validation_reconstruction_mse": validation_reconstruction,
            "validation_total_loss": validation_total / len(validation_indices),
            "validation_code_statistics": statistics,
            "validation_quality_gate": gate,
        }
        epoch_log.append(record)
        print(
            f"epoch={epoch} train_total={record['train_total_loss']:.6f} "
            f"val_reconstruction_mse={validation_reconstruction:.6f} "
            f"used_codes={[row['used_codes'] for row in statistics['codebook_usage_by_level']]} "
            f"quality_pass={gate['passed']}",
            flush=True,
        )
        score = checkpoint_score(statistics, gate, validation_reconstruction)
        if best_score is None or score > best_score:
            best_score = score
            best_loss = validation_reconstruction
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            selected_quality = {"statistics": statistics, "gate": gate}
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
        warmup_log=warmup_log,
        selected_validation_quality=selected_quality,
        initialization=initialization,
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
    if model.quantizer.update_method == "ema":
        metrics["ema_training_updates"] = int(model.quantizer.ema_updates)
        metrics["dead_code_resets_by_level"] = model.quantizer.dead_resets.cpu().tolist()
    mapping.metadata = {
        "model_type": "trained_rqvae",
        "config": asdict(model.config),
        "collision_policy": "lookup returns every catalog item with the generated code",
    }
    return mapping, metrics
