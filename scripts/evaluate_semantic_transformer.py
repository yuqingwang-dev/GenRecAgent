"""Evaluate a trained next-code Transformer through the shared valid-item adapter."""

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

from genrecagent.data.m3_audit import resolve_project_output
from genrecagent.data.storage import load_histories, read_jsonl
from genrecagent.evaluation.generative import evaluate_generative
from genrecagent.retrieval.generative import GenerativeRetriever, train_popularity_counts
from genrecagent.retrieval.transformer import (
    CausalSemanticTransformer,
    SemanticVocabulary,
    TransformerTokenScorer,
)
from genrecagent.semantic_id.codebook import SemanticCodebook

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/generative.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sequences-dir", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--semantic-ids", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=["validation", "test", "both"], default="validation")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--max-codes", type=int)
    parser.add_argument("--max-users", type=int, help="Explicit evaluation smoke subset")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test != (args.max_users is not None):
        parser.error("--smoke-test and --max-users must be given together")
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = settings["source"]
    sequences = args.sequences_dir or ROOT / source["output_path"]
    prepared = args.prepared_dir or ROOT / source["prepared_path"]
    semantic_ids = args.semantic_ids or ROOT / source["semantic_ids_path"]
    checkpoint = args.checkpoint or ROOT / settings["transformer"]["output_path"] / "transformer.pt"
    output = resolve_project_output(
        ROOT, args.output_dir or Path(settings["transformer"].get(
            "evaluation_path", "outputs/m3_transformer_evaluation"
        ))
    )
    source_manifest = json.loads((sequences / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_config = json.loads(
        (checkpoint.parent / "run_config.json").read_text(encoding="utf-8")
    )
    checkpoint_metrics = json.loads(
        (checkpoint.parent / "metrics.json").read_text(encoding="utf-8")
    )
    code_hash = _sha256(semantic_ids)
    if code_hash != source_manifest["semantic_ids_sha256"]:
        raise ValueError("Semantic ID mapping differs from sequence preparation")
    if checkpoint_config["semantic_ids_sha256"] != code_hash:
        raise ValueError("Checkpoint was trained with a different Semantic ID mapping")
    if _sha256(sequences / "train.jsonl") != checkpoint_config["train_examples_sha256"]:
        raise ValueError("Checkpoint training data does not match the current M3 examples")
    if _sha256(prepared / "histories.jsonl") != source_manifest["histories_sha256"]:
        raise ValueError("Prepared histories do not match the M3 examples")
    if not args.smoke_test and checkpoint_metrics["status"] != "full_train_prefixes":
        raise ValueError("A smoke/subset checkpoint cannot be reported as a full evaluation")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = perf_counter()
    codebook = SemanticCodebook.load(semantic_ids)
    model, vocabulary, max_history = CausalSemanticTransformer.load(checkpoint)
    if vocabulary.tokens != SemanticVocabulary.from_codebook(codebook).tokens:
        raise ValueError("Checkpoint token vocabulary differs from the item catalog")
    if max_history != source_manifest["max_history"]:
        raise ValueError("Checkpoint history window differs from prepared examples")
    scorer = TransformerTokenScorer(
        model, vocabulary, max_history=max_history, device=device
    )
    histories = load_histories(prepared)
    beam_size = (
        args.beam_size if args.beam_size is not None else settings["retrieval"]["beam_size"]
    )
    max_codes = (
        args.max_codes if args.max_codes is not None else settings["retrieval"]["max_codes"]
    )
    retriever = GenerativeRetriever(
        codebook,
        scorer,
        train_popularity=train_popularity_counts(histories),
        beam_size=beam_size,
        max_codes=max_codes,
    )
    train_rows = list(read_jsonl(sequences / "train.jsonl"))
    target_codes = {tuple(row["target_code"]) for row in train_rows}
    splits = ("validation", "test") if args.split == "both" else (args.split,)
    results = {}
    for split in splits:
        rows = list(read_jsonl(sequences / f"{split}.jsonl"))
        if args.max_users is not None:
            if not 1 <= args.max_users <= len(rows):
                raise ValueError("max_users must be in the split range")
            rows = rows[:args.max_users]
        results[split] = evaluate_generative(
            rows, retriever, list(settings["evaluation"]["k_values"]),
            target_codes, split,
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip())
    run_config = {
        "status": "smoke_test" if args.smoke_test else "full_evaluation",
        "command": " ".join(sys.argv),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_training_config": checkpoint_config,
        "dataset_manifest": source_manifest,
        "retrieval": {"beam_size": beam_size, "max_codes": max_codes},
        "evaluation": settings["evaluation"],
        "device": device,
        "latency_scope": (
            "Transformer inference, trie search, collision expansion and seen filtering"
        ),
        "git_commit": commit,
        "git_dirty": dirty,
        "platform": platform.platform(),
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Output: {output}; elapsed: {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
