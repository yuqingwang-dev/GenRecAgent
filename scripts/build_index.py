"""Encode static catalog metadata and build an exact inner-product index."""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.data.amazon import build_item_text
from genrecagent.data.storage import load_items
from genrecagent.retrieval.dense import DenseRetriever
from genrecagent.retrieval.encoder import SentenceItemEncoder, TextEncoder

ROOT = Path(__file__).resolve().parents[1]


def _catalog_ids(item_ids: list[str], max_items: int | None, seed: int) -> list[str]:
    if max_items is None:
        return item_ids
    if max_items < 1:
        raise ValueError("max_items must be positive")
    return sorted(
        item_ids,
        key=lambda item_id: hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest(),
    )[:max_items]


def build_catalog_index(
    items_path: Path,
    encoder: TextEncoder,
    *,
    batch_size: int,
    backend: str,
    max_items: int | None,
    seed: int,
    dataset_manifest: dict[str, object],
) -> DenseRetriever:
    items = load_items(items_path)
    item_ids = _catalog_ids(sorted(items), max_items, seed)
    texts = [build_item_text(items[item_id]) for item_id in item_ids]
    if not all(texts):
        raise ValueError("Catalog contains an item with no usable metadata text")
    vectors = encoder.encode(texts, batch_size)
    index = DenseRetriever(backend=backend)
    index.build_index(
        item_ids,
        vectors,
        metadata={
            "dataset": dataset_manifest.get("dataset"),
            "category": dataset_manifest.get("category"),
            "dataset_revision": dataset_manifest.get("revision"),
            "review_sha256": dataset_manifest.get("review_sha256"),
            "metadata_sha256": dataset_manifest.get("metadata_sha256"),
            "model_name": encoder.model_name,
            "device": encoder.device,
            "batch_size": batch_size,
            "catalog_sampling": "full" if max_items is None else "SHA256(seed:item_id)",
            "max_items": max_items,
            "seed": seed,
            "created_utc": datetime.now(UTC).isoformat(),
        },
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/retrieval.yaml")
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--backend", choices=["numpy", "faiss"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prepared = args.prepared_dir or ROOT / config["dataset"]["prepared_path"]
    output = args.output_dir or ROOT / config["retrieval"]["index_path"]
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    seed = int(manifest["stats"]["seed"])
    encoder_config = config["encoder"]
    if encoder_config["normalize_embeddings"] is not True:
        raise ValueError("M1 exact cosine baseline requires normalized item embeddings")
    started = perf_counter()
    print(f"Dataset: {manifest['dataset']} / {manifest['category']}; seed: {seed}")
    print(f"Users: {manifest['stats']['users']}; interactions: {manifest['stats']['interactions']}")
    print(
        f"Model: {encoder_config['model_name']}; "
        f"device: {args.device or encoder_config['device']}"
    )
    encoder = SentenceItemEncoder(
        encoder_config["model_name"],
        cache_dir=ROOT / encoder_config["cache_dir"],
        device=args.device or encoder_config["device"],
        allow_download=args.allow_model_download,
    )
    index = build_catalog_index(
        prepared,
        encoder,
        batch_size=int(encoder_config["batch_size"]),
        backend=args.backend or config["retrieval"]["backend"],
        max_items=args.max_items,
        seed=seed,
        dataset_manifest=manifest,
    )
    index.metadata["elapsed_seconds"] = round(perf_counter() - started, 3)
    index.save_index(output)
    print(f"Items: {len(index.item_ids)}; dimension: {index.embeddings.shape[1]}")
    print(f"Backend: {index.backend}; normalization: l2; device: {encoder.device}")
    print(f"Output: {output}; elapsed: {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
