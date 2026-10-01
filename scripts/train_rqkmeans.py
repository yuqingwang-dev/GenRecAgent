"""Fit a residual MiniBatch K-Means Semantic ID baseline on M1 item vectors."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.semantic_id.codebook import SemanticCodebook
from genrecagent.semantic_id.projection import PCAWhitening, PCAWhiteningConfig
from genrecagent.semantic_id.rqkmeans import ResidualKMeans, RQKMeansConfig
from genrecagent.semantic_id.training import split_item_indices

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
    if not 2 <= max_items <= len(item_ids):
        raise ValueError("max_items must be between two and the catalog size")
    ordered = sorted(
        range(len(item_ids)),
        key=lambda index: (
            hashlib.sha256(f"{seed}:{item_ids[index]}".encode()).digest(), item_ids[index]
        ),
    )
    return np.asarray(ordered[:max_items], dtype=np.int64)


def _override(value: int | None, default: int) -> int:
    return default if value is None else value


def _usage(codes: np.ndarray, size: int) -> list[dict[str, float | int]]:
    usage = []
    for level in range(codes.shape[1]):
        counts = np.bincount(codes[:, level], minlength=size)
        probabilities = counts[counts > 0] / max(1, counts.sum())
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        used = int(np.count_nonzero(counts))
        usage.append({
            "used_codes": used,
            "codebook_size": size,
            "utilization": used / size,
            "perplexity": math.exp(entropy),
        })
    return usage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rqkmeans.yaml")
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--num-levels", type=int)
    parser.add_argument("--codebook-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--projection-dim", type=int)
    parser.add_argument("--no-projection", action="store_true")
    args = parser.parse_args()
    if args.smoke_test and args.max_items is None:
        parser.error("--smoke-test requires --max-items")
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    index_dir = args.index_dir or ROOT / settings["source"]["index_path"]
    output_dir = args.output_dir or ROOT / settings["source"]["output_path"]
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {output_dir}")
    embedding_path = index_dir / "item_embeddings.npy"
    ids_path = index_dir / "item_ids.json"
    vectors = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    all_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    source = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    if vectors.ndim != 2 or vectors.shape[0] != len(all_ids):
        raise ValueError("Index embeddings and item IDs do not align")
    if source["normalization"] != "l2":
        raise ValueError("RQ-KMeans baseline expects L2-normalized embeddings")
    seed = int(settings["model"]["seed"])
    selected = _sample_indices(all_ids, args.max_items, seed)
    item_ids = [all_ids[index] for index in selected]
    selected_vectors = vectors if args.max_items is None else np.asarray(vectors[selected])
    train_indices, validation_indices = split_item_indices(
        item_ids,
        seed=seed,
        validation_fraction=float(settings["evaluation"]["validation_fraction"]),
    )
    started = perf_counter()
    projection_settings = settings["projection"]
    projection_enabled = projection_settings["enabled"] and not args.no_projection
    projector: PCAWhitening | None = None
    if projection_enabled:
        projection_dim = _override(args.projection_dim, projection_settings["output_dim"])
        projector = PCAWhitening(
            PCAWhiteningConfig(input_dim=int(vectors.shape[1]),
                               output_dim=projection_dim, seed=seed)
        ).fit(np.asarray(selected_vectors[train_indices]))
        model_vectors = projector.transform(selected_vectors)
    else:
        model_vectors = np.asarray(selected_vectors, dtype=np.float32)
    model_settings = settings["model"]
    config = RQKMeansConfig(
        input_dim=int(model_vectors.shape[1]),
        num_levels=_override(args.num_levels, model_settings["num_levels"]),
        codebook_size=_override(args.codebook_size, model_settings["codebook_size"]),
        batch_size=_override(args.batch_size, model_settings["batch_size"]),
        max_iter=_override(args.max_iter, model_settings["max_iter"]),
        seed=seed,
    )
    print(
        f"RQ-KMeans baseline: {len(item_ids)} items, input {vectors.shape[1]} "
        f"-> {config.input_dim} dimensions; "
        f"train={len(train_indices)} validation={len(validation_indices)}",
        flush=True,
    )
    model = ResidualKMeans(config).fit(model_vectors[train_indices])
    validation_codes, validation_residual = model.encode(model_vectors[validation_indices])
    codes, residual = model.encode(model_vectors)
    validation_reconstruction = model.reconstruct(validation_codes)
    catalog_reconstruction = model.reconstruct(codes)
    if projector is not None:
        validation_reconstruction = projector.inverse_transform(validation_reconstruction)
        catalog_reconstruction = projector.inverse_transform(catalog_reconstruction)
        pca_only_validation = float(np.mean((
            np.asarray(selected_vectors[validation_indices])
            - projector.inverse_transform(model_vectors[validation_indices])
        ) ** 2))
    else:
        pca_only_validation = 0.0
    mapping = SemanticCodebook()
    for item_id, row in zip(item_ids, codes, strict=True):
        mapping.add(item_id, tuple(f"<s{level}:{int(code)}>" for level, code in enumerate(row)))
    status = (
        "cpu_smoke" if args.smoke_test
        else "development_subset" if args.max_items is not None
        else "full_catalog_rqkmeans_baseline"
    )
    mapping.metadata = {
        "model_type": "rq_kmeans_baseline",
        "run_status": status,
        "config": asdict(config),
        "projection": asdict(projector.config) if projector else None,
        "source_dataset": source["dataset"],
        "category": source["category"],
        "dataset_revision": source["dataset_revision"],
        "source_model_name": source["model_name"],
        "source_model_revision": source.get("model_revision"),
        "item_embedding_sha256": _sha256(embedding_path),
        "collision_policy": "lookup returns all catalog items sharing a learned code",
    }
    metrics = {
        "status": status,
        "item_count": len(item_ids),
        "train_item_count": len(train_indices),
        "validation_item_count": len(validation_indices),
        "validation_reconstruction_mse": float(np.mean((
            np.asarray(selected_vectors[validation_indices]) - validation_reconstruction
        ) ** 2)),
        "catalog_reconstruction_mse": float(np.mean((
            np.asarray(selected_vectors) - catalog_reconstruction
        ) ** 2)),
        "validation_projected_residual_mse": float(np.mean(validation_residual**2)),
        "catalog_projected_residual_mse": float(np.mean(residual**2)),
        "pca_only_validation_reconstruction_mse": pca_only_validation,
        "codebook_usage_by_level": _usage(codes, config.codebook_size),
        "collisions": mapping.collision_stats(),
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True,
        check=False,
    ).stdout.strip())
    run_config = {
        "command": " ".join(sys.argv),
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "git_commit": commit,
        "git_dirty": dirty,
        "source_index_metadata": source,
        "item_embedding_sha256": mapping.metadata["item_embedding_sha256"],
        "item_ids_sha256": _sha256(ids_path),
        "model": asdict(config),
        "projection": asdict(projector.config) if projector else None,
        "selected_item_count": len(item_ids),
        "subset_sampling": "full" if args.max_items is None else "SHA256(seed:item_id)",
        "numpy_version": np.__version__,
        "scikit_learn_version": version("scikit-learn"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "rqkmeans_codebooks.npz")
    if projector is not None:
        projector.save(output_dir / "pca_whitening.npz")
    mapping.save(output_dir / "item_semantic_ids.json")
    for name, payload in (
        ("metrics.json", metrics),
        ("fit_log.json", model.fit_log),
        ("run_config.json", run_config),
        ("train_item_ids.json", [item_ids[index] for index in train_indices]),
        ("validation_item_ids.json", [item_ids[index] for index in validation_indices]),
    ):
        (output_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Output: {output_dir}; elapsed: {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
