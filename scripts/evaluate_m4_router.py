"""Evaluate the dense-first offline router on validation histories only."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.agent.offline_factory import load_offline_router
from genrecagent.agent.offline_router import OfflineRouteRequest
from genrecagent.evaluation.recommendation import ndcg_at_k, recall_at_k

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path,
                        default=ROOT / "data/processed/All_Beauty_full_optimized")
    parser.add_argument("--index-dir", type=Path,
                        default=ROOT / "data/processed/All_Beauty/index")
    parser.add_argument("--m3-sequences-dir", type=Path,
                        default=ROOT / "data/processed/All_Beauty_m3")
    parser.add_argument("--semantic-ids", type=Path,
                        default=ROOT / "outputs/m2_rqkmeans_full/item_semantic_ids.json")
    parser.add_argument("--generative-config", type=Path,
                        default=ROOT / "configs/generative.yaml")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/m4_router_validation")
    parser.add_argument("--mode", choices=["auto", "dense", "generative"], default="auto")
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test != (args.max_users is not None):
        parser.error("--smoke-test and --max-users must be given together")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    enable_generative = args.mode == "generative"
    config = yaml.safe_load(args.generative_config.read_text(encoding="utf-8")) if (
        enable_generative
    ) else None
    started = perf_counter()
    resources = load_offline_router(
        args.prepared_dir, args.index_dir,
        enable_generative=enable_generative,
        m3_sequences_dir=args.m3_sequences_dir if enable_generative else None,
        semantic_ids_path=args.semantic_ids if enable_generative else None,
        transition_alpha=float(config["transition"]["alpha"]) if config else 10.0,
        beam_size=int(config["retrieval"]["beam_size"]) if config else 20,
        max_codes=int(config["retrieval"]["max_codes"]) if config else 20,
    )
    histories = resources.histories
    if args.max_users is not None:
        if not 1 <= args.max_users <= len(histories):
            raise ValueError("max-users must lie within the validation split")
        histories = histories[:args.max_users]
    k_values = (5, 10, 20)
    recall: dict[int, list[float]] = {k: [] for k in k_values}
    ndcg: dict[int, list[float]] = {k: [] for k in k_values}
    route_counts: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    fallback_count = failed_count = unreachable = target_in_history = 0
    latencies: list[float] = []
    attempt_counts: list[int] = []
    catalog = resources.router.catalog
    for number, user in enumerate(histories, start=1):
        history = user.validation_history
        target = user.validation_item
        unreachable += target not in catalog
        target_in_history += target in history
        call_started = perf_counter()
        result = resources.router.route(OfflineRouteRequest(
            tuple(history), top_k=max(k_values), mode=args.mode
        ))
        latencies.append((perf_counter() - call_started) * 1000)
        attempt_counts.append(len(result.attempts))
        route_counts[result.selected_route or "none"] += 1
        fallback_count += result.fallback_used
        failed_count += not result.satisfied
        for attempt in result.attempts:
            if not attempt.success:
                failure_reasons[attempt.reason or "unknown"] += 1
        predicted = [hit.item_id for hit in result.hits]
        for k in k_values:
            recall[k].append(recall_at_k(predicted, {target}, k))
            ndcg[k].append(ndcg_at_k(predicted, {target}, k))
        if number % 500 == 0:
            print(f"validation: routed {number}/{len(histories)} users", flush=True)
    metrics = {
        "status": "smoke_test" if args.smoke_test else "full_validation_routing",
        "split": "validation",
        "requested_mode": args.mode,
        "users": len(histories),
        "items": len(catalog),
        "unreachable_targets": unreachable,
        "target_already_in_history": target_in_history,
        "route_counts": dict(route_counts),
        "fallback_count": fallback_count,
        "failed_count": failed_count,
        "attempt_failure_reasons": dict(failure_reasons),
        "avg_attempts": mean(attempt_counts),
        "avg_route_latency_ms": mean(latencies),
        **{f"Recall@{k}": mean(recall[k]) for k in k_values},
        **{f"NDCG@{k}": mean(ndcg[k]) for k in k_values},
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
        "command": " ".join(sys.argv),
        "status": metrics["status"],
        "split": "validation_only",
        "policy": "dense_first_v1; explicit generative only; popularity cold-start/fallback",
        "prepared_manifest": resources.prepared_manifest,
        "index_metadata": resources.router.dense.metadata,
        "histories_sha256": _sha256(args.prepared_dir / "histories.jsonl"),
        "m3_manifest": resources.m3_manifest,
        "m3_train_examples_sha256": resources.train_examples_sha256,
        "candidate_universe": "full static item metadata catalog",
        "latency_scope": "routing, retrieval and hit verification; excludes loading",
        "git_commit": commit,
        "git_dirty": dirty,
        "platform": platform.platform(),
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
    print(json.dumps(metrics, indent=2))
    print(f"Output: {args.output_dir}; elapsed: {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
