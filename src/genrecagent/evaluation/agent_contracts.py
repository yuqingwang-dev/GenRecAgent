"""Independent output grading and explicitly defined agent metric denominators."""

from math import isfinite

from genrecagent.agent.constraints import PriceLimit


def legal_action_attempt(step: dict) -> bool:
    return (
        step["observation"]["ok"]
        or step.get("service_failure", False)
        or "synthetic_injected_tool_failure" in step["observation"].get("error", "")
    )


def grade_contract(result: dict, task: dict, items: dict) -> dict:
    rows = result["items"]
    ids = [row["item_id"] for row in rows]
    price = PriceLimit(task["max_price"], inclusive=task.get("price_inclusive", True))
    checks = {
        "catalog": all(item_id in items for item_id in ids),
        "unique": len(ids) == len(set(ids)),
        "history_exclusion": not (set(ids) & set(task["history"])),
        "price": all(item_id in items and price.allows(items[item_id].price) for item_id in ids),
        "category": all(
            item_id in items
            and (task["category"] is None or task["category"] in items[item_id].categories)
            for item_id in ids
        ),
        "ranks": all(row.get("rank") == index for index, row in enumerate(rows, 1)),
        "scores": all(
            isinstance(row.get("score"), (float, int))
            and not isinstance(row.get("score"), bool)
            and isfinite(row["score"])
            for row in rows
        ),
    }
    audited_pool = None
    for step in result["trace"]:
        audit = step.get("candidate_audit")
        if audit and audit["kind"] == "filtered_pools":
            audited_pool = {hit["item_id"] for pool in audit["sources"].values() for hit in pool}
    provenance = (
        all(item_id in audited_pool for item_id in ids) if audited_pool is not None else None
    )
    safe = all(checks.values()) and provenance is not False
    correct = safe and result["status"] == task["expected"]
    if result["status"] == "success":
        correct = correct and len(rows) == task["top_k"]
    elif result["status"] == "clarification":
        correct = correct and not rows
    return {
        "contract_pass": bool(correct),
        "output_safe": bool(safe),
        "checks": checks,
        "provenance_checked": provenance,
        "nonempty_output_safe": bool(safe) if rows else None,
    }


def end_reason(result: dict, *, max_steps: int = 10) -> str:
    status = result["status"]
    last = result["trace"][-1]["observation"] if result["trace"] else {}
    if status == "success":
        return (
            "termination_gate_bypassed"
            if last.get("gate_removed_for_ablation")
            else "verified_count_fulfilled"
        )
    if status == "clarification":
        return last.get("reason", "clarification_without_recorded_reason")
    if status == "budget_exhausted":
        return (
            "step_budget_exhausted"
            if result["steps"] >= max_steps
            else "execution_deadline_exhausted"
        )
    if status == "tool_failure":
        return "tool_failure_without_recovery"
    return "unclassified_terminal_status"


def summarize_contracts(traces: list[dict]) -> dict:
    if not traces:
        raise ValueError("Agent metric denominators need actual tasks")
    decisions, illegal, calls, successful = 0, 0, 0, 0
    nonempty = safe_nonempty = feasible = fills = infeasible = clarifications = 0
    for row in traces:
        task, result, grade = row["task"], row["result"], row["grade"]
        if result["items"]:
            nonempty += 1
            safe_nonempty += grade["nonempty_output_safe"] is True
        if task["expected"] == "success":
            feasible += 1
            fills += grade["contract_pass"] and result["status"] == "success"
        elif task["expected"] == "clarification":
            infeasible += 1
            clarifications += grade["contract_pass"] and result["status"] == "clarification"
        for index, step in enumerate(result["trace"]):
            if task["scenario"] == "early_finish" and index == 0:
                continue  # Externally seeded action, not an agent decision.
            decisions += 1
            illegal += not legal_action_attempt(step)
            if step["action"]["name"] != "policy_error":
                calls += 1
                successful += step["observation"]["ok"]
    return {
        "agent_decision_attempts": decisions,
        "legal_action_attempt_rate": (decisions - illegal) / decisions if decisions else None,
        "tool_call_attempts": calls,
        "mean_tool_call_attempts": calls / len(traces),
        "successful_tool_calls": successful,
        "mean_successful_tool_calls": successful / len(traces),
        "nonempty_results": nonempty,
        "hard_constraint_satisfaction_rate_on_nonempty": safe_nonempty / nonempty
        if nonempty
        else None,
        "feasible_fulfillment_rate": fills / feasible if feasible else None,
        "correct_no_result_rate": clarifications / infeasible if infeasible else None,
        "definitions": {
            "legal_action_attempt_rate": "excludes seeded early finish; "
            "validated backend failure is a legal call",
            "tool_call_attempts": "agent trace entries excluding seeded action "
            "and JSON/policy decode errors",
            "constraint_rate": "only nonempty outputs; "
            "empty-output systems do not receive a 100% score",
        },
    }
