"""Raw validation selection: no runtime guard, test tasks, or next-action labels."""

import hashlib

from genrecagent.agent.compact_policy import CompactQwenStepPolicy
from genrecagent.agent.constraints import PriceLimit
from genrecagent.agent.tool_environment import ToolEnvironment, ToolTask, run_tool_agent
from genrecagent.evaluation.agent_contracts import grade_contract, legal_action_attempt


def validation_subset(rows: list[dict], *, count: int = 16) -> list[dict]:
    if count < 2 or count % 2:
        raise ValueError("Use an even positive validation subset size")
    if any(row["split"] != "validation" for row in rows):
        raise ValueError("Checkpoint selection may only read validation tasks")
    selected = []
    for outcome in ("success", "clarification"):
        group = sorted(
            (row for row in rows if row["expected"] == outcome),
            key=lambda row: hashlib.sha256(f"42:selection:{row['task_id']}".encode()).hexdigest(),
        )
        if len(group) < count // 2:
            raise ValueError("Insufficient stratified validation tasks")
        selected += group[: count // 2]
    return selected


def evaluate_raw_rollouts(llm, resources, items: dict, rows: list[dict]) -> dict:
    policy = CompactQwenStepPolicy(llm)
    passed, filled, errors, tokens = 0, 0, 0, 0
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
        env = ToolEnvironment(resources, items, task)
        if row["scenario"] == "early_finish":
            env.step({"name": "finish", "arguments": {}})
        result = run_tool_agent(env, policy)
        grade = grade_contract(result, row, items)
        passed += grade["contract_pass"]
        filled += grade["contract_pass"] and result["status"] == "success"
        for index, step in enumerate(result["trace"]):
            errors += not legal_action_attempt(step) and not (
                row["scenario"] == "early_finish" and index == 0
            )
            tokens += (step.get("policy_generation") or {}).get("output_tokens", 0)
    return {
        "tasks": len(rows),
        "contract_pass": passed,
        "feasible_filled": filled,
        "policy_illegal_calls": errors,
        "generation_tokens": tokens,
        "task_ids": [row["task_id"] for row in rows],
        "controller_actions": 0,
    }


def selection_score(rollout: dict, nll: float) -> tuple:
    return (
        rollout["feasible_filled"],
        rollout["contract_pass"],
        -rollout["policy_illegal_calls"],
        -nll,
    )
