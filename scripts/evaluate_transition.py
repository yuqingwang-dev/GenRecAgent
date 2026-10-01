"""Fit a train-only code-transition baseline and evaluate valid item retrieval."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.data.storage import load_histories, read_jsonl
from genrecagent.evaluation.generative import evaluate_generative as _evaluate
from genrecagent.retrieval.generative import GenerativeRetriever, train_popularity_counts
from genrecagent.retrieval.transition import CodeTransitionScorer
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
    parser.add_argument("--sequences-dir", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--semantic-ids", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=["validation", "test", "both"], default="validation")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = config["source"]
    sequences = args.sequences_dir or ROOT / source["output_path"]
    prepared = args.prepared_dir or ROOT / source["prepared_path"]
    semantic_ids = args.semantic_ids or ROOT / source["semantic_ids_path"]
    output = args.output_dir or ROOT / config["evaluation"]["output_path"]
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    sequence_manifest = json.loads((sequences / "manifest.json").read_text(encoding="utf-8"))
    if _sha256(semantic_ids) != sequence_manifest["semantic_ids_sha256"]:
        raise ValueError("Semantic ID mapping differs from sequence preparation")
    if _sha256(prepared / "histories.jsonl") != sequence_manifest["histories_sha256"]:
        raise ValueError("History data differs from sequence preparation")
    started = perf_counter()
    codebook = SemanticCodebook.load(semantic_ids)
    histories = load_histories(prepared)
    training_rows = list(read_jsonl(sequences / "train.jsonl"))
    scorer = CodeTransitionScorer(
        codebook, alpha=float(config["transition"]["alpha"])
    ).fit(training_rows)
    retriever = GenerativeRetriever(
        codebook,
        scorer,
        train_popularity=train_popularity_counts(histories),
        beam_size=int(config["retrieval"]["beam_size"]),
        max_codes=int(config["retrieval"]["max_codes"]),
    )
    target_codes = {tuple(row["target_code"]) for row in training_rows}
    splits = ("validation", "test") if args.split == "both" else (args.split,)
    results = {
        split: _evaluate(
            list(read_jsonl(sequences / f"{split}.jsonl")),
            retriever,
            list(config["evaluation"]["k_values"]),
            target_codes,
            split,
        )
        for split in splits
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip())
    run_config = {
        "model_type": "train_only_code_transition_baseline",
        "command": " ".join(sys.argv),
        "dataset_manifest": sequence_manifest,
        "semantic_ids_sha256": _sha256(semantic_ids),
        "train_examples_sha256": _sha256(sequences / "train.jsonl"),
        "train_examples": len(training_rows),
        "retrieval": config["retrieval"],
        "transition": config["transition"],
        "evaluation": config["evaluation"],
        "candidate_universe": "full static item metadata catalog",
        "collision_tie_breaker": "train-interaction popularity then item ID",
        "latency_scope": "code scoring, trie decoding, collision expansion and seen filtering",
        "git_commit": commit,
        "git_dirty": dirty,
        "platform": platform.platform(),
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    output.mkdir(parents=True, exist_ok=True)
    scorer.save(output / "transition_model.json")
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Output: {output}; elapsed: {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
