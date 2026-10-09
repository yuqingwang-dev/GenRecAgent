"""Train a small causal next-Semantic-ID Transformer on M3 train prefixes."""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.data.m3_audit import resolve_project_output

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _override(value: int | None, default: int) -> int:
    return default if value is None else value


def _subset(rows: list[dict], count: int | None, seed: int) -> list[dict]:
    if count is None:
        return rows
    if not 1 <= count <= len(rows):
        raise ValueError("Subset size must be between one and the split size")
    import numpy as np

    selected = np.random.default_rng(seed).choice(len(rows), size=count, replace=False)
    return [rows[int(index)] for index in sorted(selected)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/generative.yaml")
    parser.add_argument("--sequences-dir", type=Path)
    parser.add_argument("--semantic-ids", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--feedforward-dim", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--approved", action="store_true",
                        help="Use only after explicit approval of an approval-required config")
    args = parser.parse_args()
    if args.smoke_test and (
        args.max_train_examples is None or args.max_validation_examples is None
    ):
        parser.error("--smoke-test needs explicit train and validation subset sizes")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["transformer"].get("approval_required") and not args.approved:
        parser.error("This experiment requires separate training approval; no training started")
    source = config["source"]
    sequences = args.sequences_dir or ROOT / source["output_path"]
    semantic_ids = args.semantic_ids or ROOT / source["semantic_ids_path"]
    output = resolve_project_output(
        ROOT, args.output_dir or Path(config["transformer"]["output_path"])
    )
    sequence_manifest = json.loads((sequences / "manifest.json").read_text(encoding="utf-8"))
    if _sha256(semantic_ids) != sequence_manifest["semantic_ids_sha256"]:
        raise ValueError("Semantic ID mapping differs from the prepared M3 samples")
    expected_hash = source.get("expected_semantic_ids_sha256")
    if expected_hash and _sha256(semantic_ids) != expected_hash:
        raise ValueError("Semantic ID mapping differs from the frozen experiment")
    for split in ("train", "validation"):
        recorded = sequence_manifest.get("example_file_sha256", {}).get(split)
        if recorded and _sha256(sequences / f"{split}.jsonl") != recorded:
            raise ValueError(f"Prepared sample hash mismatch: {split}")

    import torch

    from genrecagent.data.storage import read_jsonl
    from genrecagent.retrieval.transformer import SemanticVocabulary, TransformerConfig
    from genrecagent.retrieval.transformer_training import (
        TransformerTrainingConfig,
        fit_semantic_transformer,
    )
    from genrecagent.semantic_id.codebook import SemanticCodebook

    codebook = SemanticCodebook.load(semantic_ids)
    vocabulary = SemanticVocabulary.from_codebook(codebook)
    max_history = int(sequence_manifest["max_history"])
    settings = config["transformer"]
    model_config = TransformerConfig(
        vocab_size=vocabulary.size,
        max_positions=max_history * 4 + 3,
        pad_id=vocabulary.pad_id,
        d_model=_override(args.d_model, settings["d_model"]),
        n_heads=int(settings["n_heads"]),
        n_layers=_override(args.n_layers, settings["n_layers"]),
        feedforward_dim=_override(args.feedforward_dim, settings["feedforward_dim"]),
        dropout=float(settings["dropout"]),
    )
    train_config = TransformerTrainingConfig(
        epochs=_override(args.epochs, settings["epochs"]),
        batch_size=_override(args.batch_size, settings["batch_size"]),
        learning_rate=float(settings["learning_rate"]),
        seed=int(settings["seed"]),
        device=args.device or settings["device"],
        gradient_clip=float(settings["gradient_clip"]),
    )
    train_rows = _subset(
        list(read_jsonl(sequences / "train.jsonl")),
        args.max_train_examples,
        train_config.seed,
    )
    validation_rows = _subset(
        list(read_jsonl(sequences / "validation.jsonl")),
        args.max_validation_examples,
        train_config.seed + 1,
    )
    started = perf_counter()
    print(
        f"Transformer: {len(train_rows)} train prefixes, {len(validation_rows)} "
        f"validation examples, vocab={vocabulary.size}, device={train_config.device}",
        flush=True,
    )
    result = fit_semantic_transformer(
        train_rows,
        validation_rows,
        vocabulary,
        model_config,
        train_config,
        max_history=max_history,
    )
    status = "cpu_smoke" if args.smoke_test and result.device == "cpu" else (
        "smoke_test" if args.smoke_test else "development_subset"
        if args.max_train_examples is not None else "full_train_prefixes"
    )
    best_record = result.epoch_log[result.best_epoch - 1]
    metrics = {
        "status": status,
        "best_epoch": result.best_epoch,
        "best_validation_target_nll": result.best_validation_loss,
        "best_validation_token_accuracy": best_record["validation_token_accuracy"],
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "supervised_target_tokens_per_example": 3,
        "parameters": sum(parameter.numel() for parameter in result.model.parameters()),
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip())
    run_config = {
        "command": " ".join(sys.argv),
        "dataset_manifest": sequence_manifest,
        "semantic_ids_sha256": _sha256(semantic_ids),
        "train_examples_sha256": _sha256(sequences / "train.jsonl"),
        "validation_examples_sha256": _sha256(sequences / "validation.jsonl"),
        "model": asdict(model_config),
        "training": asdict(train_config),
        "max_history": max_history,
        "actual_device": result.device,
        "git_commit": commit,
        "git_dirty": dirty,
        "torch_version": torch.__version__,
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "config_sha256": _sha256(args.config),
        "explicit_training_approval_flag": args.approved,
        "source_hashes": {
            name: _sha256(ROOT / name) for name in (
                "scripts/train_semantic_transformer.py",
                "src/genrecagent/retrieval/transformer.py",
                "src/genrecagent/retrieval/transformer_training.py",
            )
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    result.model.save(output / "transformer.pt", vocabulary, max_history)
    (output / "vocabulary.json").write_text(
        json.dumps(list(vocabulary.tokens)), encoding="utf-8"
    )
    for name, payload in (
        ("training_log.json", result.epoch_log),
        ("metrics.json", metrics),
        ("run_config.json", run_config),
    ):
        (output / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Output: {output}; elapsed: {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
