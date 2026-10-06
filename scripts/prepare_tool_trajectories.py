"""Execute synthetic teacher tasks and export disjoint multi-step SFT samples."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.agent.constraints import PriceLimit
from genrecagent.agent.tool_environment import (
    FixedStepPolicy,
    ToolEnvironment,
    ToolTask,
    run_tool_agent,
)
from genrecagent.agent.trajectory_data import action_tokens, step_messages
from genrecagent.data.storage import load_items
from genrecagent.retrieval.hybrid_loader import file_sha256, load_dual_resources

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/tool_sft_v2")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output must be empty")
    from transformers import AutoTokenizer

    model_config = json.loads((ROOT / "outputs/m4_qwen_download/manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["snapshot_path"], local_files_only=True, trust_remote_code=False
    )
    resources = load_dual_resources(ROOT)
    items = load_items(ROOT / "data/processed/All_Beauty_full_optimized")
    users = sorted(
        resources.base.histories,
        key=lambda user: hashlib.sha256(f"42:tool-sft:{user.user_id}".encode()).hexdigest(),
    )
    budgets = {
        "train": (0.0, 5.0, 10.0, 20.0),
        "validation": (0.05, 7.0, 15.0),
        "test": (0.1, 12.0, 25.0),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests, offset, maximum_tokens = {}, 0, 0
    for split, limits in budgets.items():
        tasks, pairs, traces = [], [], []
        for limit in limits:
            for mode in ("dense", "hybrid", "generative", "popularity"):
                for category in (None, "Premium Beauty"):
                    user = users[offset]
                    offset += 1
                    group = f"{split}:{mode}:{limit}:{category}"
                    cases = [("known", user.train_items, ()), ("cold", (), ())]
                    if mode in {"dense", "hybrid"}:
                        cases.append(("tool_failure", user.train_items, ("retrieve_dense",)))
                        cases.append(("early_finish", user.train_items, ()))
                    for scenario, history, faults in cases:
                        text = f"Recommend using {mode}; each item at most ${limit}."
                        if category:
                            text += f" Catalog category must be {category}."
                        task = ToolTask(
                            f"{group}:{scenario}",
                            text,
                            tuple(history),
                            mode=mode,
                            top_k=5,
                            price_limit=PriceLimit(limit),
                            category=category,
                            injected_failures=frozenset(faults),
                        )
                        environment = ToolEnvironment(resources, items, task)
                        if scenario == "early_finish":
                            environment.step({"name": "finish", "arguments": {}})
                            environment.trace[-1]["supervise"] = False
                        result = run_tool_agent(environment, FixedStepPolicy())
                        feasible = environment.eligible_count >= task.top_k
                        if (feasible and result["status"] != "success") or (
                            not feasible and result["status"] != "clarification"
                        ):
                            raise ValueError(
                                f"Teacher task failed independent feasibility check: {task.task_id}"
                            )
                        for hit in result["items"]:
                            item = items[hit["item_id"]]
                            if (
                                item.price is None
                                or item.price > limit
                                or item.item_id in history
                                or (category is not None and category not in item.categories)
                            ):
                                raise ValueError("Teacher produced an invalid item")
                        task_record = {
                            "task_id": task.task_id,
                            "split": split,
                            "group": group,
                            "user_group": user.user_id,
                            "text": text,
                            "history": list(history),
                            "mode": mode,
                            "top_k": task.top_k,
                            "max_price": limit,
                            "category": category,
                            "faults": list(faults),
                            "scenario": scenario,
                            "expected": "success" if feasible else "clarification",
                        }
                        tasks.append(task_record)
                        traces.append({"task": task_record, "result": result})
                        for step in result["trace"]:
                            if step.get("supervise", True):
                                messages = step_messages(step["state"], step["action"])
                                encoded = action_tokens(tokenizer, messages)
                                maximum_tokens = max(maximum_tokens, len(encoded["input_ids"]))
                                pairs.append(
                                    {
                                        "task_id": task.task_id,
                                        "group": group,
                                        "source": "synthetic_rule_teacher_executed_on_real_catalog",
                                        "messages": messages,
                                    }
                                )
        for suffix, rows in (("tasks", tasks), ("actions", pairs), ("traces", traces)):
            path = args.output_dir / f"{split}_{suffix}.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifests[split] = {
            "tasks": len(tasks),
            "assistant_actions": len(pairs),
            "outcomes": dict(Counter(row["expected"] for row in tasks)),
            "task_sha256": file_sha256(args.output_dir / f"{split}_tasks.jsonl"),
            "actions_sha256": file_sha256(args.output_dir / f"{split}_actions.jsonl"),
            "user_groups": sorted({row["user_group"] for row in tasks}),
        }
    if any(
        set(manifests[a]["user_groups"]) & set(manifests[b]["user_groups"])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("User group leakage")
    manifest = {
        "source": "synthetic, actual tool execution; not online user trajectories",
        "split_rule": "disjoint users and intent/price/category goal combinations; prices disjoint",
        "budgets": budgets,
        "splits": manifests,
        "max_sequence_tokens": maximum_tokens,
        "base_model": model_config["model_name"],
        "base_revision": model_config["revision"],
        "semantic_checkpoint_sha256": resources.checkpoint_sha256,
        "assistant_only_labels": True,
        "observation_snapshots_immutable": True,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "splits"}, indent=2))
    print(
        json.dumps(
            {
                key: {k: v for k, v in value.items() if k != "user_groups"}
                for key, value in manifests.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
