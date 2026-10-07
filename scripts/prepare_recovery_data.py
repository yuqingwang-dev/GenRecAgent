"""Execute training-only correction snippets and new disjoint validation/test task families."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.agent.compact_policy import PROTOCOL, compact_messages
from genrecagent.agent.constraints import PriceLimit
from genrecagent.agent.guarded_policy import allowed_actions, controller_action
from genrecagent.agent.recovery_data import (
    TEST_TEMPLATES,
    TRAIN_TEMPLATES,
    VALIDATION_TEMPLATES,
    balanced_actions,
    check_split_groups,
)
from genrecagent.agent.tool_environment import ToolEnvironment, ToolTask, run_tool_agent
from genrecagent.agent.trajectory_data import action_tokens
from genrecagent.data.storage import load_items, read_jsonl
from genrecagent.delivery.demo import grade_demo
from genrecagent.retrieval.hybrid_loader import file_sha256, load_dual_resources

ROOT = Path(__file__).resolve().parents[1]


def hash_number(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest(), 16)


class Teacher:
    def choose(self, state):
        return controller_action(state, allowed_actions(state))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/processed/tool_sft_recovery_v3"
    )
    parser.add_argument("--per-action", type=int, default=120)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output must be fresh")
    if not torch.cuda.is_available():
        parser.error(
            "CUDA required for existing retrieval artifacts; no Qwen training is performed"
        )
    from transformers import AutoTokenizer

    original = ROOT / "data/processed/tool_sft_v2"
    prior = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT
        / "artifacts/model_cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"
        / prior["base_revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    resources = load_dual_resources(ROOT)
    items = load_items(ROOT / "data/processed/All_Beauty_full_optimized")
    reserved = set().union(*(set(info["user_groups"]) for info in prior["splits"].values()))
    users = sorted(
        (u for u in resources.base.histories if u.user_id not in reserved),
        key=lambda u: hash_number(f"42:recovery-v3:{u.user_id}"),
    )
    records = {"train": [], "validation": [], "test": []}
    for source in read_jsonl(original / "train_tasks.jsonl"):
        row = {**source, "task_id": "v3:" + source["task_id"]}
        row["text"] = TRAIN_TEMPLATES[hash_number(row["task_id"]) % len(TRAIN_TEMPLATES)].format(
            mode=row["mode"], price=row["max_price"], top_k=row["top_k"]
        )
        records["train"].append(row)
    prices = {
        "train": (0.01, 0.03, 0.2, 0.5, 1.0, 3.0, 11.0, 21.0),
        "validation": (0.06, 8.0, 18.0),
        "test": (0.07, 6.0, 19.0),
    }
    templates = {
        "train": TRAIN_TEMPLATES,
        "validation": VALIDATION_TEMPLATES,
        "test": TEST_TEMPLATES,
    }
    offset = 0
    for split, budgets in prices.items():
        for price in budgets:
            for mode in ("dense", "hybrid", "generative", "popularity"):
                for category in (None, "Premium Beauty"):
                    user = users[offset]
                    offset += 1
                    cases = [("known", user.train_items, []), ("cold", (), [])]
                    if mode in {"dense", "hybrid"}:
                        cases += [
                            ("tool_failure", user.train_items, ["retrieve_dense"]),
                            ("early_finish", user.train_items, []),
                        ]
                    for scenario, history, faults in cases:
                        task_id = f"{split}:{mode}:{price}:{category}:{scenario}"
                        text = templates[split][
                            hash_number(task_id) % len(templates[split])
                        ].format(mode=mode, price=price, top_k=5)
                        if category:
                            text += f" Catalog category: {category}."
                        records[split].append(
                            {
                                "task_id": task_id,
                                "split": split,
                                "user_group": user.user_id,
                                "group": f"{split}:{mode}:{price}:{category}",
                                "text": text,
                                "history": list(history),
                                "mode": mode,
                                "top_k": 5,
                                "max_price": price,
                                "category": category,
                                "faults": faults,
                                "scenario": scenario,
                            }
                        )
    check_split_groups(records)
    heldout = {row["user_group"] for split in ("validation", "test") for row in records[split]}
    if heldout & reserved:
        raise ValueError("Fresh held-out users appeared in any v2 split")
    actions, traces = {s: [] for s in records}, {s: [] for s in records}

    def environment(row, *, faults=True):
        return ToolEnvironment(
            resources,
            items,
            ToolTask(
                row["task_id"],
                row["text"],
                tuple(row["history"]),
                mode=row["mode"],
                top_k=row["top_k"],
                price_limit=PriceLimit(row["max_price"]),
                category=row["category"],
                injected_failures=frozenset(row["faults"] if faults else []),
            ),
        )

    def sample(row, step, suffix):
        return {
            "sample_id": f"{row['task_id']}:{suffix}",
            "split": row["split"],
            "task_id": row["task_id"],
            "user_group": row["user_group"],
            "group": row["group"],
            "source": "synthetic_task_actual_catalog_execution",
            "state": step["state"],
            "action": step["action"],
            "observation": step["observation"],
            "executed_action_checked": step["observation"]["ok"],
            "messages": compact_messages(step["state"], step["action"]),
        }

    for split, rows in records.items():
        for number, row in enumerate(rows):
            env = environment(row)
            if row["scenario"] == "early_finish":
                env.step({"name": "finish", "arguments": {}})
                env.trace[-1]["supervise"] = False
            result = run_tool_agent(env, Teacher())
            feasible = env.eligible_count >= row["top_k"]
            row["expected"] = "success" if feasible else "clarification"
            # Independent metadata check including the exact category.
            eligible = {
                key: item
                for key, item in items.items()
                if row["category"] is None or row["category"] in item.categories
            }
            grade = grade_demo(
                result,
                eligible,
                history=row["history"],
                top_k=row["top_k"],
                max_price=row["max_price"],
            )
            if not grade["contract_pass"]:
                raise ValueError(f"Teacher failed actual outcome grading: {row['task_id']}")
            traces[split].append({"task": row, "result": result, "grade": grade})
            for index, step in enumerate(result["trace"]):
                if step.get("supervise", True) and step["observation"]["ok"]:
                    actions[split].append(sample(row, step, f"normal:{index}"))
            # Error recovery is collected ONLY from training goals, never held-out outputs.
            if split == "train" and feasible:
                repair = environment(row, faults=False)
                prefix = hash_number(row["task_id"]) % 3
                for _ in range(prefix):
                    repair.step(Teacher().choose(repair.snapshot()))
                bad = (
                    {"name": "filter_candidates", "arguments": {}}
                    if prefix == 0
                    else {"name": "retrieve_dense", "arguments": {"top_k": True}}
                    if prefix == 1
                    else {"name": "finish", "arguments": {}}
                )
                repetitions = 1 + hash_number(row["task_id"] + ":repeat") % 2
                for _ in range(repetitions):
                    if repair.step(bad)["ok"]:
                        raise ValueError("Expected an actually rejected injected action")
                call = Teacher().choose(repair.snapshot())
                if not repair.step(call)["ok"]:
                    raise ValueError("Corrective action was not physically accepted")
                actions["train"].append(sample(row, repair.trace[-1], "checked_partial_repair"))
                traces["train"].append(
                    {
                        "task_id": row["task_id"],
                        "source": "partial_error_repair",
                        "scope": "next action checked, not a claimed completed task",
                        "trace": repair.trace,
                    }
                )
            if number % 40 == 0:
                print(f"{split}: {number + 1}/{len(rows)} executed", flush=True)
    pool_count = len(actions["train"])
    actions["train"], balance = balanced_actions(actions["train"], per_action=args.per_action)
    maximum = 0
    for rows in actions.values():
        for row in rows:
            encoded = action_tokens(tokenizer, row["messages"], action_name_weight=4.0)
            maximum = max(maximum, len(encoded["input_ids"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for split in records:
        for name, rows in (
            ("tasks", records[split]),
            ("actions", actions[split]),
            ("traces", traces[split]),
        ):
            with (args.output_dir / f"{split}_{name}.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        info[split] = {
            "tasks": len(records[split]),
            "assistant_actions": len(actions[split]),
            "outcomes": dict(Counter(row["expected"] for row in records[split])),
            "user_groups": sorted({row["user_group"] for row in records[split]}),
            "actions_sha256": file_sha256(args.output_dir / f"{split}_actions.jsonl"),
            "task_sha256": file_sha256(args.output_dir / f"{split}_tasks.jsonl"),
        }
    manifest = {
        "source": "synthetic tasks; real tool execution; train-only partial correction snippets",
        "input_protocol": PROTOCOL,
        "base_revision": prior["base_revision"],
        "base_model": prior["base_model"],
        "splits": info,
        "observation_snapshots_immutable": True,
        "assistant_only_labels": True,
        "action_name_weight": 4.0,
        "max_sequence_tokens": maximum,
        "candidate_train_actions": pool_count,
        "balanced_sampling": balance,
        "templates": templates,
        "price_sets": {s: sorted({r["max_price"] for r in rows}) for s, rows in records.items()},
        "fresh_heldout_users_exclude_all_v2_groups": True,
        "semantic_checkpoint_sha256": resources.checkpoint_sha256,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "splits": {
                    s: {k: v for k, v in r.items() if k != "user_groups"} for s, r in info.items()
                },
                "max_tokens": maximum,
                "balanced_sampling": balance,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
