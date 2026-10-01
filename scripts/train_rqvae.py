"""Train RQ-VAE on saved M1 item embeddings and encode the catalog."""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.semantic_id.rqvae import RQVAEConfig
from genrecagent.semantic_id.training import TrainingConfig, encode_catalog, fit_rqvae

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_indices(item_ids: list[str], max_items: int | None, seed: int) -> np.ndarray:
    if max_items is None:
        return np.arange(len(item_ids), dtype=np.int64)
    if max_items < 2:
        raise ValueError("max_items must be at least two")
    if max_items > len(item_ids):
        raise ValueError("max_items exceeds the available catalog size")
    ordered = sorted(
        range(len(item_ids)),
        key=lambda index: (
            hashlib.sha256(f"{seed}:{item_ids[index]}".encode()).digest(), item_ids[index]
        ),
    )
    return np.asarray(ordered[:max_items], dtype=np.int64)


def _override(value: int | None, default: int) -> int:
    return default if value is None else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/semantic_id.yaml")
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-items", type=int, help="Explicit smoke/development subset")
    parser.add_argument("--smoke-test", action="store_true", help="Label this run as a smoke test")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--num-levels", type=int)
    parser.add_argument("--codebook-size", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--init-samples", type=int)
    args = parser.parse_args()
    if args.smoke_test and args.max_items is None:
        parser.error("--smoke-test requires --max-items")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    index_dir = args.index_dir or ROOT / config["source"]["index_path"]
    output_dir = args.output_dir or ROOT / config["source"]["output_path"]
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {output_dir}")
    embedding_path = index_dir / "item_embeddings.npy"
    item_ids_path = index_dir / "item_ids.json"
    vectors = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    all_item_ids = json.loads(item_ids_path.read_text(encoding="utf-8"))
    source_metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    if vectors.ndim != 2 or vectors.shape[0] != len(all_item_ids):
        raise ValueError("Index embeddings and item IDs do not align")
    if source_metadata["normalization"] != "l2":
        raise ValueError("RQ-VAE M2 expects L2-normalized M1 embeddings")
    seed = int(config["training"]["seed"])
    selected = _sample_indices(all_item_ids, args.max_items, seed)
    item_ids = [all_item_ids[index] for index in selected]
    training_vectors = vectors if args.max_items is None else np.asarray(vectors[selected])
    model_config = RQVAEConfig(
        input_dim=int(vectors.shape[1]),
        hidden_dim=_override(args.hidden_dim, config["model"]["hidden_dim"]),
        latent_dim=_override(args.latent_dim, config["model"]["latent_dim"]),
        num_levels=_override(args.num_levels, config["model"]["num_levels"]),
        codebook_size=_override(args.codebook_size, config["model"]["codebook_size"]),
        commitment_weight=float(config["model"]["commitment_weight"]),
    )
    train_config = TrainingConfig(
        epochs=_override(args.epochs, config["training"]["epochs"]),
        batch_size=_override(args.batch_size, config["training"]["batch_size"]),
        learning_rate=float(config["training"]["learning_rate"]),
        validation_fraction=float(config["training"]["validation_fraction"]),
        seed=seed,
        device=args.device or config["training"]["device"],
        init_samples=(
            args.init_samples
            if args.init_samples is not None
            else config["training"]["init_samples"]
        ),
    )
    started = perf_counter()
    print(
        f"RQ-VAE input: {len(item_ids)} items x {vectors.shape[1]} dimensions; "
        f"seed={seed}, device={train_config.device}",
        flush=True,
    )
    result = fit_rqvae(training_vectors, item_ids, model_config, train_config)
    mapping, catalog_metrics = encode_catalog(
        result.model,
        training_vectors,
        item_ids,
        batch_size=train_config.batch_size,
        device=result.device,
    )
    mapping.metadata.update(
        {
            "source_dataset": source_metadata["dataset"],
            "category": source_metadata["category"],
            "dataset_revision": source_metadata["dataset_revision"],
            "item_embedding_sha256": _sha256(embedding_path),
            "source_model_name": source_metadata["model_name"],
            "source_model_revision": source_metadata.get("model_revision"),
            "seed": seed,
            "item_count": len(item_ids),
        }
    )
    status = (
        "cpu_smoke" if args.smoke_test and result.device == "cpu"
        else "smoke_test" if args.smoke_test
        else "development_subset" if args.max_items is not None
        else "full_catalog_training"
    )
    metrics = {
        "status": status,
        "best_epoch": result.best_epoch,
        "best_validation_reconstruction_mse": result.best_validation_reconstruction_mse,
        "train_item_count": len(result.train_indices),
        "validation_item_count": len(result.validation_indices),
        **catalog_metrics,
    }
    mapping.metadata["run_status"] = status
    mapping.metadata["train_item_count"] = len(result.train_indices)
    mapping.metadata["validation_item_count"] = len(result.validation_indices)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=False
    ).stdout.strip())
    run_config = {
        "command": " ".join(sys.argv),
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "git_commit": commit,
        "git_dirty": dirty,
        "source_index_metadata": source_metadata,
        "item_embedding_sha256": mapping.metadata["item_embedding_sha256"],
        "item_ids_sha256": _sha256(item_ids_path),
        "model": asdict(model_config),
        "training": asdict(train_config),
        "selected_item_count": len(item_ids),
        "subset_sampling": "full" if args.max_items is None else "SHA256(seed:item_id)",
        "codebook_initialization": "train_latent_kmeans_plus_plus_seeds",
        "actual_device": result.device,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result.model.save(output_dir / "rqvae.pt")
    mapping.save(output_dir / "item_semantic_ids.json")
    (output_dir / "train_item_ids.json").write_text(
        json.dumps([item_ids[index] for index in result.train_indices]), encoding="utf-8"
    )
    (output_dir / "validation_item_ids.json").write_text(
        json.dumps([item_ids[index] for index in result.validation_indices]), encoding="utf-8"
    )
    for filename, payload in (
        ("training_log.json", result.epoch_log),
        ("metrics.json", metrics),
        ("run_config.json", run_config),
    ):
        (output_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Output: {output_dir}; elapsed: {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
