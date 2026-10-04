"""Validation-only comparison of fixed code-rank fusion weights."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.data.storage import load_histories, read_jsonl
from genrecagent.evaluation.diagnostics import RetrievalBucket
from genrecagent.retrieval.generative import (
    GenerativeRetriever,
    expand_code_predictions,
    train_popularity_counts,
)
from genrecagent.retrieval.rank_fusion import fuse_code_ranks
from genrecagent.retrieval.transformer import (
    CausalSemanticTransformer,
    SemanticVocabulary,
    TransformerTokenScorer,
)
from genrecagent.retrieval.transition import CodeTransitionScorer
from genrecagent.semantic_id.codebook import SemanticCodebook

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = (0.1, 0.25, 0.5, 0.75)
RANK_OFFSET = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/generative.yaml")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "outputs/m3_transformer_full_server/transformer.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/m3_rank_fusion_validation")
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
        raise ValueError("Semantic ID mapping differs from samples or checkpoint")
    if train_hash != checkpoint_config["train_examples_sha256"]:
        raise ValueError("Train examples differ from the checkpoint")
    if _sha256(prepared / "histories.jsonl") != manifest["histories_sha256"]:
        raise ValueError("Prepared histories differ from the M3 samples")
    if checkpoint_metrics["status"] != "full_train_prefixes":
        raise ValueError("A subset checkpoint cannot be used for full validation")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = perf_counter()
    codebook = SemanticCodebook.load(semantic_ids)
    model, vocabulary, max_history = CausalSemanticTransformer.load(args.checkpoint)
    if vocabulary.tokens != SemanticVocabulary.from_codebook(codebook).tokens:
        raise ValueError("Checkpoint vocabulary differs from catalog")
    if max_history != manifest["max_history"]:
        raise ValueError("Checkpoint history window differs from samples")
    train_rows = list(read_jsonl(sequences / "train.jsonl"))
    rows = list(read_jsonl(sequences / "validation.jsonl"))
    if args.max_users is not None:
        if not 1 <= args.max_users <= len(rows):
            raise ValueError("max-users must lie within validation split")
        rows = rows[:args.max_users]
    histories = load_histories(prepared)
    popularity = train_popularity_counts(histories)
    beam_size = int(config["retrieval"]["beam_size"])
    max_codes = int(config["retrieval"]["max_codes"])
    transition = GenerativeRetriever(
        codebook,
        CodeTransitionScorer(codebook, alpha=float(config["transition"]["alpha"]))
        .fit(train_rows),
        train_popularity=popularity, beam_size=beam_size, max_codes=max_codes,
    )
    transformer = GenerativeRetriever(
        codebook,
        TransformerTokenScorer(model, vocabulary, max_history=max_history, device=device),
        train_popularity=popularity, beam_size=beam_size, max_codes=max_codes,
    )
    buckets = {
        "transition": RetrievalBucket(), "transformer": RetrievalBucket(),
        **{f"transformer_weight_{weight:g}": RetrievalBucket() for weight in WEIGHTS},
    }
    diagnostics = {name: Counter() for name in buckets}
    union_code_found = 0
    for number, row in enumerate(rows, start=1):
        target = row["target_item"]
        target_code = tuple(row["target_code"])
        history = row["history_items"]
        outcomes = {
            "transition": transition.retrieve(history, top_k=20),
            "transformer": transformer.retrieve(history, top_k=20),
        }
        union_code_found += target_code in {
            prediction.semantic_id
            for outcome in outcomes.values() for prediction in outcome.predicted_codes
        }
        for weight in WEIGHTS:
            name = f"transformer_weight_{weight:g}"
            fused = fuse_code_ranks(
                outcomes["transition"].predicted_codes,
                outcomes["transformer"].predicted_codes,
                transformer_weight=weight, max_codes=max_codes,
                rank_offset=RANK_OFFSET,
            )
            outcomes[name] = expand_code_predictions(
                fused, codebook, train_popularity=popularity,
                seen_items=set(history), top_k=20,
            )
        for name, outcome in outcomes.items():
            buckets[name].add(target, target_code, outcome)
            diagnostics[name].update(asdict(outcome.diagnostics))
        if number % 500 == 0:
            print(f"fusion: evaluated {number}/{len(rows)} validation users", flush=True)
    summaries = {
        name: {**bucket.summary(), "diagnostic_totals": dict(diagnostics[name])}
        for name, bucket in buckets.items()
    }
    candidates = ["transition", *(f"transformer_weight_{weight:g}" for weight in WEIGHTS)]
    selected = max(
        candidates,
        key=lambda name: (summaries[name]["Recall@20"], summaries[name]["NDCG@20"]),
    )
    metrics = {
        "validation_users": len(rows),
        "oracle_union_code_recall_at_20_each": union_code_found / len(rows),
        "methods": summaries,
        "selected_by_validation_recall20_then_ndcg20": selected,
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
        "status": "smoke_test" if args.smoke_test else "full_validation_fusion_grid",
        "split": "validation_only",
        "command": " ".join(sys.argv),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "train_examples_sha256": train_hash,
        "validation_examples_sha256": _sha256(sequences / "validation.jsonl"),
        "semantic_ids_sha256": code_hash,
        "dataset_manifest": manifest,
        "weights": WEIGHTS,
        "rank_offset": RANK_OFFSET,
        "beam_size": beam_size,
        "max_codes": max_codes,
        "device": device,
        "platform": platform.platform(),
        "git_commit": commit,
        "git_dirty": dirty,
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "selected": selected,
        "oracle_union_code_recall": metrics["oracle_union_code_recall_at_20_each"],
        "Recall@20": {name: row["Recall@20"] for name, row in summaries.items()},
    }, indent=2))
    print(f"Output: {args.output_dir}; elapsed: {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
