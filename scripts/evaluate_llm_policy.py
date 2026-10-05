"""Measure raw Qwen decisions, guarded execution and rules on frozen task labels."""

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

from genrecagent.agent.constrained_agent import ConstrainedRecommendationAgent
from genrecagent.agent.offline_factory import load_offline_router
from genrecagent.agent.tool_policy import PROMPT_SHA256, SCHEMA_SHA256, LLMToolPolicy
from genrecagent.data.storage import load_items, read_jsonl
from genrecagent.evaluation.constraint_tasks import grade_constraint_task
from genrecagent.evaluation.task_summary import summarize_task_records
from genrecagent.llm.local_qwen import LocalQwenLLM

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _raw_correct(task: dict, decision) -> bool:
    parsed = decision.attempts[0].parsed
    if parsed is None:
        return False
    payload = parsed.to_dict()
    return all(payload[key] == task["expected"][key] for key in ("status", "intent", "price_limit"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path,
                        default=ROOT / "data/processed/All_Beauty_m4_tasks")
    parser.add_argument("--split", choices=["development", "heldout"], default="development")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/llm_policy.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-templates", type=int)
    args = parser.parse_args()
    if args.smoke_test != (args.max_templates is not None):
        parser.error("--smoke-test and --max-templates must be given together")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    manifest = json.loads((args.tasks_dir / "manifest.json").read_text(encoding="utf-8"))
    task_file = args.tasks_dir / f"{args.split}.jsonl"
    prepared = ROOT / "data/processed/All_Beauty_full_optimized"
    if _sha256(task_file) != manifest["split_sha256"][args.split]:
        raise ValueError("Task file differs from the frozen manifest")
    if _sha256(prepared / "items.jsonl") != manifest["items_sha256"] or (
        _sha256(prepared / "histories.jsonl") != manifest["histories_sha256"]
    ):
        raise ValueError("Prepared source differs from the task set")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    download = json.loads((ROOT / "outputs/m4_qwen_download/manifest.json").read_text(
        encoding="utf-8"
    ))
    if download["revision"] != config["revision"] or (
        download["files"]["model.safetensors"]["sha256"] != config["weights_sha256"]
    ):
        raise ValueError("Downloaded weights differ from the pinned policy configuration")
    tasks = list(read_jsonl(task_file))
    if any(task["split"] != args.split for task in tasks):
        raise ValueError("Task split mismatch")
    if args.max_templates is not None:
        templates = sorted({task["template_id"] for task in tasks})
        if not 1 <= args.max_templates <= len(templates):
            raise ValueError("Invalid template subset")
        selected = set(templates[:args.max_templates])
        tasks = [task for task in tasks if task["template_id"] in selected]
    started = perf_counter()
    llm = LocalQwenLLM(
        **{key: config[key] for key in (
            "model_name", "revision", "device", "dtype", "max_input_tokens", "max_new_tokens"
        )}, cache_dir=ROOT / config["cache_dir"], allow_download=False,
    )
    resources = load_offline_router(
        prepared, ROOT / "data/processed/All_Beauty/index", enable_generative=True,
        m3_sequences_dir=ROOT / "data/processed/All_Beauty_m3",
        semantic_ids_path=ROOT / "outputs/m2_rqkmeans_full/item_semantic_ids.json",
    )
    items = load_items(prepared)
    agent, policy = ConstrainedRecommendationAgent(resources.router, items), LLMToolPolicy(llm)
    records, decisions = [], []
    for task in tasks:
        call_started = perf_counter()
        rules_result = agent.run(task["text"], tuple(task["history_items"]), top_k=task["top_k"])
        records.append({
            "task_id": task["task_id"], "method": "rules", "expected": task["expected"],
            "result": rules_result.to_dict(),
            "grade": grade_constraint_task(task, rules_result, items),
            "latency_ms": (perf_counter() - call_started) * 1000,
        })
        call_started = perf_counter()
        decision = policy.decide(task["text"])
        result = agent.execute_intent(
            decision.parsed, tuple(task["history_items"]), top_k=task["top_k"]
        )
        decisions.append((task, decision))
        records.append({
            "task_id": task["task_id"], "method": "llm_guarded", "expected": task["expected"],
            "result": result.to_dict(), "grade": grade_constraint_task(task, result, items),
            "latency_ms": (perf_counter() - call_started) * 1000,
            "policy": decision.to_dict(), "first_raw_labels_correct": _raw_correct(task, decision),
        })
        if not decision.cache_hit:
            print(
                f"{args.split}: template {task['template_id']}, "
                f"attempts={len(decision.attempts)}, rules_fallback={decision.rules_fallback}",
                flush=True,
            )
    unique = [(task, decision) for task, decision in decisions if not decision.cache_hit]
    attempts = [attempt for _, decision in unique for attempt in decision.attempts]
    generations = [attempt.generation for attempt in attempts if attempt.generation]
    metrics = {
        "status": "smoke_test" if args.smoke_test else "full_policy_task_evaluation",
        "split": args.split, "task_set": manifest["task_set"],
        "methods": {
            method: summarize_task_records([row for row in records if row["method"] == method])
            for method in ("rules", "llm_guarded")
        },
        "policy": {
            "unique_texts": len(unique), "model_calls": len(attempts),
            "first_schema_valid_unique": sum(d.attempts[0].schema_valid for _, d in unique),
            "first_raw_labels_correct_unique": sum(_raw_correct(t, d) for t, d in unique),
            "first_raw_labels_correct_tasks": sum(_raw_correct(t, d) for t, d in decisions),
            "schema_or_provider_errors": sum(not a.schema_valid for a in attempts),
            "guard_rejections": sum(a.schema_valid and not a.guard_valid for a in attempts),
            "repaired_unique_texts": sum(len(d.attempts) > 1 for _, d in unique),
            "rules_fallback_unique": sum(d.rules_fallback for _, d in unique),
            "rules_fallback_tasks": sum(d.rules_fallback for _, d in decisions),
            "cache_hit_tasks": sum(d.cache_hit for _, d in decisions),
            "input_tokens": sum(g["input_tokens"] for g in generations),
            "output_tokens": sum(g["output_tokens"] for g in generations),
            "generation_seconds": sum(g["elapsed_seconds"] for g in generations),
        },
        "scope": "curated task labels and static metadata; raw model and guarded results separated",
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
        "command": " ".join(sys.argv), "model": llm.runtime,
        "weights_sha256": config["weights_sha256"], "task_manifest": manifest,
        "task_sha256": _sha256(task_file), "prompt_sha256": PROMPT_SHA256,
        "schema_sha256": SCHEMA_SHA256, "max_attempts": policy.max_attempts,
        "model_input_scope": "request text and capability/schema prompt; no labels/user IDs",
        "cache_scope": "text-only intent decisions reused across three history contexts",
        "m3_train_examples_sha256": resources.train_examples_sha256,
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
