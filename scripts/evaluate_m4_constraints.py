"""Evaluate frozen controlled-language tasks with independent metadata checks."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.agent.constrained_agent import ConstrainedRecommendationAgent
from genrecagent.agent.offline_factory import load_offline_router
from genrecagent.data.storage import load_items, read_jsonl
from genrecagent.evaluation.constraint_tasks import grade_constraint_task
from genrecagent.evaluation.task_summary import summarize_task_records as _summarize

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path,
                        default=ROOT / "data/processed/All_Beauty_m4_tasks")
    parser.add_argument("--split", choices=["development", "heldout"], default="development")
    parser.add_argument("--prepared-dir", type=Path,
                        default=ROOT / "data/processed/All_Beauty_full_optimized")
    parser.add_argument("--index-dir", type=Path,
                        default=ROOT / "data/processed/All_Beauty/index")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    manifest = json.loads((args.tasks_dir / "manifest.json").read_text(encoding="utf-8"))
    task_file = args.tasks_dir / f"{args.split}.jsonl"
    if _sha256(task_file) != manifest["split_sha256"][args.split]:
        raise ValueError("Task file differs from its frozen manifest")
    if _sha256(args.prepared_dir / "items.jsonl") != manifest["items_sha256"] or (
        _sha256(args.prepared_dir / "histories.jsonl") != manifest["histories_sha256"]
    ):
        raise ValueError("Task set and catalog/history evidence differ")
    started = perf_counter()
    resources = load_offline_router(
        args.prepared_dir, args.index_dir, enable_generative=True,
        m3_sequences_dir=ROOT / "data/processed/All_Beauty_m3",
        semantic_ids_path=ROOT / "outputs/m2_rqkmeans_full/item_semantic_ids.json",
    )
    items = load_items(args.prepared_dir)
    agent = ConstrainedRecommendationAgent(resources.router, items)
    tasks = list(read_jsonl(task_file))
    if not tasks or any(task["split"] != args.split for task in tasks):
        raise ValueError("Task split is empty or inconsistent")
    records = []
    for task in tasks:
        for method, prefilter in (
            ("prefilter_verified", True), ("postfilter_top_k_baseline", False)
        ):
            call_started = perf_counter()
            result = agent.run(
                task["text"], tuple(task["history_items"]), top_k=task["top_k"],
                prefilter=prefilter,
            )
            latency = (perf_counter() - call_started) * 1000
            records.append({
                "task_id": task["task_id"], "context": task["context"], "method": method,
                "text": task["text"], "expected": task["expected"],
                "result": result.to_dict(), "grade": grade_constraint_task(task, result, items),
                "latency_ms": latency,
            })
    metrics = {
        "split": args.split, "task_source": manifest["source"],
        "methods": {
            method: _summarize([row for row in records if row["method"] == method])
            for method in ("prefilter_verified", "postfilter_top_k_baseline")
        },
        "scope": "synthetic controlled-language contract checks, not recommendation relevance",
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
        "command": " ".join(sys.argv), "task_manifest": manifest,
        "task_sha256": _sha256(task_file), "split": args.split,
        "m3_train_examples_sha256": resources.train_examples_sha256,
        "index_metadata": resources.router.dense.metadata,
        "catalog_known_prices": sum(item.price is not None for item in items.values()),
        "catalog_missing_prices": sum(item.price is None for item in items.values()),
        "git_commit": commit, "git_dirty": dirty, "platform": platform.platform(),
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("metrics.json", metrics), ("run_config.json", run_config)):
        (args.output_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
