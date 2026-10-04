"""Compare Transformer and train-only transition failure modes on validation only."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.data.storage import load_histories, read_jsonl
from genrecagent.evaluation.diagnostics import RetrievalBucket, validation_cohorts
from genrecagent.retrieval.generative import GenerativeRetriever, train_popularity_counts
from genrecagent.retrieval.transformer import (
    CausalSemanticTransformer,
    SemanticVocabulary,
    TransformerTokenScorer,
)
from genrecagent.retrieval.transition import CodeTransitionScorer
from genrecagent.semantic_id.codebook import SemanticCodebook

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate_method(
    name: str, rows: list[dict], retriever: GenerativeRetriever,
    codebook: SemanticCodebook, train_target_codes: set[tuple[str, ...]],
) -> dict[str, dict[str, dict[str, int | float]]]:
    buckets: dict[str, dict[str, RetrievalBucket]] = {}
    for number, row in enumerate(rows, start=1):
        target_code = tuple(row["target_code"])
        cohorts = validation_cohorts(
            history_length=len(row["history_items"]),
            target_code_seen=target_code in train_target_codes,
            collision_size=len(codebook.lookup(target_code)),
        )
        outcome = retriever.retrieve(row["history_items"], top_k=20)
        for axis, value in cohorts:
            buckets.setdefault(axis, {}).setdefault(value, RetrievalBucket()).add(
                row["target_item"], target_code, outcome
            )
        if number % 500 == 0:
            print(f"{name}: diagnosed {number}/{len(rows)} validation users", flush=True)
    return {
        axis: {value: bucket.summary() for value, bucket in sorted(groups.items())}
        for axis, groups in buckets.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/generative.yaml")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "outputs/m3_transformer_full_server/transformer.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/m3_validation_diagnostics")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test != (args.max_users is not None):
        parser.error("--smoke-test and --max-users must be given together")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sequences = ROOT / config["source"]["output_path"]
    prepared = ROOT / config["source"]["prepared_path"]
    semantic_ids = ROOT / config["source"]["semantic_ids_path"]
    manifest = json.loads((sequences / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_config = json.loads(
        (args.checkpoint.parent / "run_config.json").read_text(encoding="utf-8")
    )
    checkpoint_metrics = json.loads(
        (args.checkpoint.parent / "metrics.json").read_text(encoding="utf-8")
    )
    code_hash = _sha256(semantic_ids)
    train_hash = _sha256(sequences / "train.jsonl")
    if code_hash != manifest["semantic_ids_sha256"] or (
        code_hash != checkpoint_config["semantic_ids_sha256"]
    ):
        raise ValueError("Semantic ID mapping differs from the M3 samples or checkpoint")
    if train_hash != checkpoint_config["train_examples_sha256"]:
        raise ValueError("Training examples differ from the checkpoint")
    if _sha256(prepared / "histories.jsonl") != manifest["histories_sha256"]:
        raise ValueError("Prepared histories differ from the M3 samples")
    if checkpoint_metrics["status"] != "full_train_prefixes":
        raise ValueError("A subset checkpoint cannot be used for this diagnostic")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = perf_counter()
    codebook = SemanticCodebook.load(semantic_ids)
    model, vocabulary, max_history = CausalSemanticTransformer.load(args.checkpoint)
    if vocabulary.tokens != SemanticVocabulary.from_codebook(codebook).tokens:
        raise ValueError("Checkpoint vocabulary differs from the catalog")
    if max_history != manifest["max_history"]:
        raise ValueError("Checkpoint history window differs from M3 samples")
    train_rows = list(read_jsonl(sequences / "train.jsonl"))
    rows = list(read_jsonl(sequences / "validation.jsonl"))
    if args.max_users is not None:
        if not 1 <= args.max_users <= len(rows):
            raise ValueError("max-users must be within the validation split")
        rows = rows[:args.max_users]
    histories = load_histories(prepared)
    popularity = train_popularity_counts(histories)
    settings = config["retrieval"]
    common = {
        "train_popularity": popularity,
        "beam_size": int(settings["beam_size"]),
        "max_codes": int(settings["max_codes"]),
    }
    retrievers = {
        "transformer": GenerativeRetriever(
            codebook,
            TransformerTokenScorer(model, vocabulary, max_history=max_history, device=device),
            **common,
        ),
        "code_transition": GenerativeRetriever(
            codebook,
            CodeTransitionScorer(
                codebook, alpha=float(config["transition"]["alpha"])
            ).fit(train_rows),
            **common,
        ),
    }
    target_codes = {tuple(row["target_code"]) for row in train_rows}
    results = {
        name: _evaluate_method(name, rows, retriever, codebook, target_codes)
        for name, retriever in retrievers.items()
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip())
    run_config = {
        "status": "smoke_test" if args.smoke_test else "full_validation_diagnostic",
        "split": "validation_only",
        "rows": len(rows),
        "train_examples": len(train_rows),
        "command": " ".join(sys.argv),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "train_examples_sha256": train_hash,
        "validation_examples_sha256": _sha256(sequences / "validation.jsonl"),
        "semantic_ids_sha256": code_hash,
        "dataset_manifest": manifest,
        "beam_size": int(settings["beam_size"]),
        "max_codes": int(settings["max_codes"]),
        "device": device,
        "platform": platform.platform(),
        "git_commit": commit,
        "git_dirty": dirty,
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        name: data["all"]["all"] for name, data in results.items()
    }, indent=2))
    print(f"Output: {args.output_dir}; elapsed: {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
