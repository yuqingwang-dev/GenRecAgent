"""Paired fixed/prompt/SFT multi-step task evaluation on a declared held-out subset."""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.agent.constraints import PriceLimit
from genrecagent.agent.guarded_policy import run_guarded_agent
from genrecagent.agent.policy_modes import GUARDED_POLICIES
from genrecagent.agent.tool_environment import (
    FixedStepPolicy,
    QwenStepPolicy,
    ToolEnvironment,
    ToolTask,
    run_tool_agent,
)
from genrecagent.data.storage import load_items, read_jsonl
from genrecagent.delivery.readiness import DEFAULT_ADAPTER, check_artifacts
from genrecagent.llm.local_qwen import LocalQwenLLM
from genrecagent.retrieval.hybrid_loader import file_sha256, load_dual_resources

ROOT = Path(__file__).resolve().parents[1]


class NoFinishVerification(ToolEnvironment):
    """Evaluation-only removal of the checked termination gate."""

    def _execute(self, name, args):
        if name == "finish":
            self.status = "success"
            return {"status": "success", "gate_removed_for_ablation": True}
        return super()._execute(name, args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", choices=["fixed", "prompt", "sft", *GUARDED_POLICIES], default="fixed"
    )
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--max-model-rejections", type=int, default=2)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-tasks", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ablation", choices=["none", "no_finish_verification", "no_recovery"], default="none"
    )
    args = parser.parse_args()
    guarded = args.policy in GUARDED_POLICIES
    sft = args.policy in {"sft", "guarded_sft"}
    if guarded and (not 1 <= args.max_model_rejections <= 10 or args.ablation != "none"):
        parser.error("Guarded policies need rejection budget [1,10] and ablation=none")
    if args.policy == "guarded_sft" and args.adapter is None:
        args.adapter = ROOT / DEFAULT_ADAPTER
    if args.adapter is not None and not sft:
        parser.error("Only an SFT policy may load an adapter")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for the cached local-model comparison")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output must be empty")
    provenance = {
        "git_commit_at_start": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes": {
            path: file_sha256(ROOT / path)
            for path in (
                "scripts/evaluate_tool_agent.py",
                "src/genrecagent/agent/tool_environment.py",
                "src/genrecagent/agent/guarded_policy.py",
                "src/genrecagent/llm/local_qwen.py",
                "configs/llm_policy.yaml",
            )
        },
        "created_utc": datetime.now(UTC).isoformat(),
        "max_steps": 10,
        "max_seconds": 120,
        "deadline_scope": "checked between actions; not a hard process timeout",
    }
    if sft:
        if args.adapter is None:
            parser.error("SFT requires a trained adapter")
        training = json.loads((args.adapter.parent / "metrics.json").read_text())
        if training["status"] != "full_sft":
            raise ValueError("A software smoke adapter is not a full SFT result")
        readiness = check_artifacts(ROOT, policy="sft", adapter=args.adapter)
        if not readiness["ready"]:
            raise ValueError("; ".join(readiness["problems"]))
    resources = load_dual_resources(ROOT)
    items = load_items(ROOT / "data/processed/All_Beauty_full_optimized")
    path = ROOT / f"data/processed/tool_sft_v2/{args.split}_tasks.jsonl"
    rows = sorted(
        read_jsonl(path), key=lambda row: hashlib.sha256(row["task_id"].encode()).hexdigest()
    )
    if not 1 <= args.max_tasks <= len(rows):
        raise ValueError("Invalid subset count")
    rows = rows[: args.max_tasks]
    policy = FixedStepPolicy()
    if args.policy != "fixed":
        config = yaml.safe_load((ROOT / "configs/llm_policy.yaml").read_text(encoding="utf-8"))
        policy = QwenStepPolicy(
            LocalQwenLLM(
                **{
                    key: config[key]
                    for key in (
                        "model_name",
                        "revision",
                        "device",
                        "dtype",
                        "max_input_tokens",
                        "max_new_tokens",
                    )
                },
                cache_dir=ROOT / config["cache_dir"],
                adapter_path=args.adapter,
            )
        )
    traces, passed, violations, latencies = [], [], 0, []
    for row in rows:
        task = ToolTask(
            row["task_id"],
            row["text"],
            tuple(row["history"]),
            mode=row["mode"],
            top_k=row["top_k"],
            price_limit=PriceLimit(row["max_price"]),
            category=row["category"],
            injected_failures=frozenset(row["faults"]),
        )
        started = perf_counter()
        environment_type = (
            NoFinishVerification if args.ablation == "no_finish_verification" else (ToolEnvironment)
        )
        environment = environment_type(resources, items, task)
        if row["scenario"] == "early_finish":
            environment.step({"name": "finish", "arguments": {}})
        result = (
            run_guarded_agent(environment, policy, max_model_rejections=args.max_model_rejections)
            if guarded
            else run_tool_agent(environment, policy, recover=args.ablation != "no_recovery")
        )
        latencies.append((perf_counter() - started) * 1000)
        item_ids = [item["item_id"] for item in result["items"]]
        safe = len(item_ids) == len(set(item_ids)) and all(
            item["item_id"] in items
            and items[item["item_id"]].price is not None
            and PriceLimit(row["max_price"]).allows(items[item["item_id"]].price)
            and item["item_id"] not in row["history"]
            and (row["category"] is None or row["category"] in items[item["item_id"]].categories)
            for item in result["items"]
        )
        violations += not safe
        correct = safe and result["status"] == row["expected"]
        if result["status"] == "success":
            correct = correct and len(result["items"]) == row["top_k"]
        elif result["status"] == "clarification":
            correct = correct and not result["items"]
        passed.append(correct)
        traces.append({"task": row, "result": result, "contract_pass": correct})
        print(
            f"{args.policy}: {row['task_id']} status={result['status']} pass={correct}", flush=True
        )
    metrics = {
        "policy": args.policy,
        "scope": f"{args.split}: full task split"
        if args.max_tasks == len(list(read_jsonl(path)))
        else f"{args.split}: fixed hash-selected held-out subset",
        "split": args.split,
        "input_scope": "normalized intent/budget/category and observed tool state",
        "tasks": len(rows),
        "contract_pass": sum(passed),
        "contract_pass_rate": mean(passed),
        "feasible_tasks": sum(row["expected"] == "success" for row in rows),
        "feasible_filled": sum(
            row["contract_pass"] and row["result"]["status"] == "success" for row in traces
        ),
        "correct_clarifications": sum(
            row["contract_pass"] and row["result"]["status"] == "clarification" for row in traces
        ),
        "constraint_violations": violations,
        "returned_items": sum(len(row["result"]["items"]) for row in traces),
        "outcomes": dict(Counter(row["result"]["status"] for row in traces)),
        "mean_steps": mean(row["result"]["steps"] for row in traces),
        "invalid_calls": sum(row["result"]["invalid_calls"] for row in traces),
        "avg_execution_latency_ms": mean(latencies),
        "generation_tokens": sum(
            (step.get("policy_generation") or {}).get("output_tokens", 0)
            for row in traces
            for step in row["result"]["trace"]
        ),
        "policy_illegal_calls": sum(
            not step["observation"]["ok"]
            and "synthetic_injected_tool_failure" not in step["observation"].get("error", "")
            and not (row["task"]["scenario"] == "early_finish" and number == 0)
            for row in traces
            for number, step in enumerate(row["result"]["trace"])
        ),
        "task_ids": [row["task_id"] for row in rows],
        "test_file_sha256": file_sha256(path) if args.split == "test" else None,
        "task_file_sha256": file_sha256(path),
        "adapter": str(args.adapter) if args.adapter else None,
        "adapter_sha256": file_sha256(args.adapter / "adapter_model.safetensors")
        if args.adapter
        else None,
        "ablation": args.ablation,
        "base_model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "recommendation_checkpoint_sha256": resources.checkpoint_sha256,
        "runtime": policy.llm.runtime if isinstance(policy, QwenStepPolicy) else None,
        "max_model_rejections": args.max_model_rejections if guarded else None,
        "guard_intervention": {
            "model_proposals": sum(
                row["result"].get("guard", {}).get("model_proposals", 0) for row in traces
            ),
            "model_rejected_proposals": sum(
                row["result"].get("guard", {}).get("model_rejected_proposals", 0) for row in traces
            ),
            "model_accepted_actions": sum(
                row["result"].get("guard", {}).get("model_accepted_actions", 0) for row in traces
            ),
            "controller_actions": sum(
                row["result"].get("guard", {}).get("controller_actions", 0) for row in traces
            ),
            "tasks_with_controller": sum(
                row["result"].get("guard", {}).get("controller_actions", 0) > 0 for row in traces
            ),
            "handoff_tasks": sum(
                row["result"].get("guard", {}).get("handoff_to_controller", False) for row in traces
            ),
            "unassisted_guarded_contract_pass": sum(
                row["contract_pass"]
                and row["result"].get("guard", {}).get("controller_actions", 0) == 0
                for row in traces
            ),
            "interpretation": "guarded system results are NOT raw-model success",
        }
        if guarded
        else None,
        **provenance,
    }
    if guarded:
        # Guard also retains generation that expired before a physical trace step could be added.
        metrics["generation_tokens"] = sum(
            row["result"]["guard"]["model_generation_tokens"] for row in traces
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as stream:
        for row in traces:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
