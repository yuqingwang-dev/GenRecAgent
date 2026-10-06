"""One-pass validation comparison of real Dense/Transformer sources and item RRF."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.evaluation.recommendation import ndcg_at_k, recall_at_k
from genrecagent.retrieval.hybrid_loader import file_sha256, load_dual_resources

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test != (args.max_users is not None):
        parser.error("--smoke-test and --max-users must be paired")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_hashes = {name: file_sha256(ROOT / name) for name in (
        "src/genrecagent/retrieval/hybrid.py", "scripts/evaluate_dual_retrieval.py"
    )}
    resources = load_dual_resources(ROOT, device=args.device)
    users = resources.base.histories
    if args.max_users is not None:
        if not 1 <= args.max_users <= len(users):
            raise ValueError("Invalid user subset")
        users = users[:args.max_users]
    weights = {"dense": 1., "generative": 0., "hybrid_50": .5, "hybrid_75": .75}
    values = {name: {f"{metric}@{k}": [] for metric in ("Recall", "NDCG") for k in (5, 10, 20)}
              for name in weights}
    latencies, failures, overlaps = [], {}, []
    for number, user in enumerate(users, start=1):
        started = perf_counter()
        outcome = resources.dual.candidates(user.validation_history, source_k=50)
        latencies.append((perf_counter() - started) * 1000)
        for source in outcome.errors:
            failures[source] = failures.get(source, 0) + 1
        overlaps.append(len({h.item_id for h in outcome.dense} & {
            h.item_id for h in outcome.generative
        }))
        for name, weight in weights.items():
            if name in {"dense", "generative"}:
                hits = outcome.dense if name == "dense" else outcome.generative
            else:
                hits = outcome.fuse(dense_weight=weight, top_k=20,
                                    seen=frozenset(user.validation_history))
            predicted = [hit.item_id for hit in hits]
            for k in (5, 10, 20):
                values[name][f"Recall@{k}"].append(
                    recall_at_k(predicted, {user.validation_item}, k)
                )
                values[name][f"NDCG@{k}"].append(ndcg_at_k(predicted, {user.validation_item}, k))
        if number % 500 == 0:
            print(f"validation: {number}/{len(users)}", flush=True)
    metrics = {
        "status": "smoke_test" if args.smoke_test else "full_validation_dual_retrieval",
        "users": len(users), "items": len(resources.dual.dense.item_ids),
        "methods": {
            name: {key: mean(v) for key, v in rows.items()} for name, rows in values.items()
        },
        "source_failures": failures, "mean_top50_overlap": mean(overlaps),
        "avg_dual_source_latency_ms": mean(latencies),
    }
    provenance = {
        "git_commit_at_start": commit, "source_hashes": source_hashes,
        "checkpoint_sha256": resources.checkpoint_sha256, "m3_manifest": resources.m3_manifest,
        "split": "validation_only", "weights_fixed_before_run": weights,
        "source_k": 50, "code_beam_and_budget": 20, "rank_offset": 20,
        "device": args.device, "created_utc": datetime.now(UTC).isoformat(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("metrics.json", metrics), ("run_config.json", provenance)):
        (args.output_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
