"""Evaluate popularity and dense next-item baselines on one catalog and split."""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.data.storage import load_histories
from genrecagent.evaluation.baseline import evaluate_baselines
from genrecagent.retrieval.dense import DenseRetriever

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/retrieval.yaml")
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=["validation", "test", "both"], default="both")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prepared = args.prepared_dir or ROOT / config["dataset"]["prepared_path"]
    index_dir = args.index_dir or ROOT / config["retrieval"]["index_path"]
    output = args.output_dir or ROOT / config["evaluation"]["output_path"]
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    started = perf_counter()
    dataset_manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    histories = load_histories(prepared)
    retriever = DenseRetriever.load_index(index_dir)
    if dataset_manifest["metadata_sha256"] != retriever.metadata["metadata_sha256"]:
        raise ValueError("Index metadata hash does not match prepared dataset")
    splits = ("validation", "test") if args.split == "both" else (args.split,)
    results = {
        split: evaluate_baselines(
            histories,
            retriever,
            split=split,
            k_values=config["evaluation"]["k_values"],
            progress_every=500,
        )
        for split in splits
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
    )
    run_config = {
        "dataset_manifest": dataset_manifest,
        "index_metadata": retriever.metadata,
        "evaluation": config["evaluation"],
        "command": "python scripts/evaluate_dense.py",
        "git_commit": commit,
        "git_dirty": dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "package_versions": {
            package: version(package)
            for package in ("numpy", "pyarrow", "torch", "sentence-transformers")
        },
        "openblas_num_threads": os.getenv("OPENBLAS_NUM_THREADS"),
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "latency_scope": "user pooling plus exact search; excludes loading and embedding model",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"Dataset: {dataset_manifest['dataset']} / {dataset_manifest['category']}")
    print(f"Source: {dataset_manifest['source_kind']}")
    print(f"Mode: {dataset_manifest['mode']}; seed: {dataset_manifest['stats']['seed']}")
    print(f"Users: {len(histories)}; items: {len(retriever.item_ids)}; "
          f"interactions: {dataset_manifest['stats']['interactions']}")
    print(f"Encoder: {retriever.metadata['model_name']}; device: {retriever.metadata['device']}")
    print(json.dumps(results, indent=2))
    print(f"Output: {output}; elapsed: {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
