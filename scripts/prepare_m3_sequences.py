"""Materialize chronological M3 examples without fitting or invoking a model."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.data.m3_audit import check_source_pins, resolve_project_output
from genrecagent.data.sequences import SequenceExample, build_sequence_examples
from genrecagent.data.storage import load_histories
from genrecagent.semantic_id.codebook import SemanticCodebook

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_examples(path: Path, examples: list[SequenceExample]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example.to_dict()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/generative.yaml")
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--semantic-ids", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-history", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = config["source"]
    prepared = args.prepared_dir or ROOT / source["prepared_path"]
    semantic_ids = args.semantic_ids or ROOT / source["semantic_ids_path"]
    output = resolve_project_output(ROOT, args.output_dir or Path(source["output_path"]))
    max_history = (
        args.max_history if args.max_history is not None else config["sequences"]["max_history"]
    )
    dataset = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    codebook = SemanticCodebook.load(semantic_ids)
    check_source_pins(source, prepared / "histories.jsonl", semantic_ids, codebook.metadata)
    if (
        codebook.metadata.get("source_dataset") != dataset["dataset"]
        or codebook.metadata.get("category") != dataset["category"]
        or codebook.metadata.get("dataset_revision") != dataset["revision"]
    ):
        raise ValueError("Semantic ID source does not match the prepared dataset")
    started = perf_counter()
    histories = load_histories(prepared)
    examples = build_sequence_examples(histories, codebook, max_history=max_history)
    counts = {split: len(rows) for split, rows in examples.items()}
    if source.get("expected_examples") and counts != source["expected_examples"]:
        raise ValueError("Example counts differ from the frozen experiment")
    without_prefix = sum(len(history.train_items) < 2 for history in histories)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip())
    manifest = {
        "dataset": dataset["dataset"],
        "category": dataset["category"],
        "dataset_revision": dataset["revision"],
        "dataset_review_sha256": dataset["review_sha256"],
        "histories_sha256": _sha256(prepared / "histories.jsonl"),
        "semantic_ids_sha256": _sha256(semantic_ids),
        "semantic_id_model_type": codebook.metadata.get("model_type"),
        "semantic_id_variant": codebook.metadata.get("variant"),
        "codebook_item_count": len(codebook.assignments),
        "max_history": max_history,
        "users": len(histories),
        "users_with_no_train_prefix": without_prefix,
        "examples": counts,
        "training_rule": "train prefix predicts a later train item only",
        "validation_rule": "train history predicts validation item",
        "test_rule": "train plus observed validation history predicts test item",
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "git_commit": commit,
        "git_dirty": dirty,
        "command": " ".join(sys.argv),
        "config_sha256": _sha256(args.config),
        "source_hashes": {
            name: _sha256(ROOT / name) for name in (
                "scripts/prepare_m3_sequences.py", "src/genrecagent/data/sequences.py",
                "src/genrecagent/data/m3_audit.py",
            )
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in examples.items():
        _write_examples(output / f"{split}.jsonl", rows)
    manifest["example_file_sha256"] = {
        split: _sha256(output / f"{split}.jsonl") for split in examples
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Output: {output}; elapsed: {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
