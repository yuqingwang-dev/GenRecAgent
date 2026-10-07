"""Offline consistency checks for recorded agent evidence; no model/tool execution."""

from collections import Counter
from math import isfinite
from statistics import mean

from genrecagent.agent.constraints import PriceLimit
from genrecagent.evaluation.agent_contracts import (
    end_reason,
    grade_contract,
    legal_action_attempt,
    summarize_contracts,
)

SOURCES = {
    "retrieve_dense": "dense",
    "retrieve_generative": "generative",
    "retrieve_popular": "popularity",
}


def audit_candidate_chain(result: dict, task: dict, items: dict) -> int:
    """Require complete successful-tool audits and a trace-consistent filtering chain."""
    pools, filtered, audits = {}, False, 0
    price = PriceLimit(task["max_price"], inclusive=task.get("price_inclusive", True))
    for step in result["trace"]:
        name, observation = step["action"]["name"], step["observation"]
        audit = step.get("candidate_audit")
        if not observation["ok"]:
            if audit is not None:
                raise ValueError("Failed tool cannot provide a successful candidate audit")
            continue
        if name in SOURCES:
            source = SOURCES[name]
            if not audit or audit.get("kind") != "retrieved_pool" or audit["source"] != source:
                raise ValueError("Successful retrieval lacks its full source audit")
            hits = audit["hits"]
            if observation["source"] != source or observation["count"] != len(hits):
                raise ValueError("Retrieval observation and full pool count disagree")
            if observation["items"] != hits[:3]:
                raise ValueError("Model-visible preview and candidate audit disagree")
            if len(hits) > step["action"]["arguments"].get("top_k", 50):
                raise ValueError("Audited retrieval exceeds its requested count")
            ids = [hit["item_id"] for hit in hits]
            if len(ids) != len(set(ids)) or any(item_id not in items for item_id in ids):
                raise ValueError("Retrieved pool has duplicates or unknown catalog IDs")
            if set(ids) & set(task["history"]):
                raise ValueError("Retrieved pool includes a seen item")
            if any(not isfinite(hit["score"]) for hit in hits):
                raise ValueError("Retrieved pool has a nonfinite score")
            pools[source], filtered = hits, False
        elif name == "filter_candidates":
            if not audit or audit.get("kind") != "filtered_pools":
                raise ValueError("Successful filter lacks its full pool audit")
            expected = {}
            for source, hits in pools.items():
                eligible = [
                    hit
                    for hit in hits
                    if price.allows(items[hit["item_id"]].price)
                    and (
                        task["category"] is None
                        or task["category"] in items[hit["item_id"]].categories
                    )
                    and hit["item_id"] not in task["history"]
                ]
                expected[source] = [{**hit, "rank": rank} for rank, hit in enumerate(eligible, 1)]
            if audit["sources"] != expected:
                raise ValueError("Filtered pool does not follow recorded retrieval and metadata")
            if observation["source_counts"] != {key: len(rows) for key, rows in expected.items()}:
                raise ValueError("Filtered observation count and audit disagree")
            pools, filtered = expected, True
        elif audit is not None:
            raise ValueError("Candidate audit is attached to an unrelated tool")
        if audit is not None:
            audits += 1
    if result["items"] and not filtered:
        raise ValueError("Nonempty output lacks a current full filtered pool")
    return audits


def audit_run(metrics: dict, rows: list[dict], tasks: list[dict], items: dict) -> dict:
    """Reject inconsistent task labels, grades, candidate provenance, or summary rates."""
    task_map = {task["task_id"]: task for task in tasks}
    if len(task_map) != len(tasks):
        raise ValueError("Original task file has duplicate task IDs")
    ids = [row["task"]["task_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)) or len(rows) != metrics["tasks"]:
        raise ValueError("Missing/duplicate result tasks or task count mismatch")
    if ids != metrics["task_ids"]:
        raise ValueError("Metrics and trace task IDs/order disagree")
    if "full task split" in metrics["scope"] and set(ids) != set(task_map):
        raise ValueError("Claimed full split omits original tasks")
    audits, provenance, grades = 0, 0, []
    for row in rows:
        task, result = row["task"], row["result"]
        if task != task_map.get(task["task_id"]):
            raise ValueError("Trace task differs from the original frozen task")
        if result["task_id"] != task["task_id"] or result["steps"] != len(result["trace"]):
            raise ValueError("Result task ID or step count mismatch")
        if result["steps"] > metrics["max_steps"]:
            raise ValueError("Recorded execution exceeds its step budget")
        if result["invalid_calls"] != sum(
            not step["observation"]["ok"] for step in result["trace"]
        ):
            raise ValueError("Result failed-call count disagrees with its trace")
        audits += audit_candidate_chain(result, task, items)
        grade = grade_contract(result, task, items)
        if grade != row["grade"] or grade["contract_pass"] != row["contract_pass"]:
            raise ValueError("Archived grade disagrees with independent metadata grading")
        if result["end_reason"] != end_reason(result, max_steps=metrics["max_steps"]):
            raise ValueError("Archived terminal reason disagrees with the trace")
        if result["items"]:
            if grade["provenance_checked"] is not True:
                raise ValueError("Nonempty output has no proven audited-pool membership")
            provenance += 1
        grades.append({**row, "grade": grade})
    contracts = sum(row["grade"]["contract_pass"] for row in grades)
    summary = summarize_contracts(grades)
    if (
        metrics["contract_pass"] != contracts
        or metrics["contract_pass_rate"] != contracts / len(rows)
        or metrics["contract_metrics"] != summary
    ):
        raise ValueError("Archived aggregate metrics disagree with the actual traces")
    counts = {
        "policy_illegal_calls": sum(
            not legal_action_attempt(step)
            and not (row["task"]["scenario"] == "early_finish" and index == 0)
            for row in rows
            for index, step in enumerate(row["result"]["trace"])
        ),
        "mean_steps": mean(row["result"]["steps"] for row in rows),
        "invalid_calls": sum(row["result"]["invalid_calls"] for row in rows),
        "feasible_tasks": sum(row["task"]["expected"] == "success" for row in rows),
        "feasible_filled": sum(
            row["contract_pass"] and row["result"]["status"] == "success" for row in rows
        ),
        "correct_clarifications": sum(
            row["contract_pass"] and row["result"]["status"] == "clarification" for row in rows
        ),
        "constraint_violations": sum(not row["grade"]["output_safe"] for row in rows),
        "returned_items": sum(len(row["result"]["items"]) for row in rows),
        "outcomes": dict(Counter(row["result"]["status"] for row in rows)),
    }
    for key, value in counts.items():
        if metrics[key] != value:
            raise ValueError(f"Archived {key} disagrees with the actual traces")
    return {
        "tasks": len(rows),
        "contract_pass": contracts,
        "candidate_audits": audits,
        "nonempty_provenance_verified": provenance,
        "contract_metrics": summary,
    }


def check_paired_runs(runs: list[tuple[dict, list[dict]]]) -> None:
    if not runs:
        raise ValueError("At least one actual run is required")
    first_metrics, first_rows = runs[0]
    for metrics, rows in runs[1:]:
        for key in (
            "split",
            "task_file_sha256",
            "base_model_revision",
            "recommendation_checkpoint_sha256",
            "max_steps",
            "max_seconds",
        ):
            if metrics[key] != first_metrics[key]:
                raise ValueError(f"Paired runs differ in {key}")
        if [row["task"] for row in rows] != [row["task"] for row in first_rows]:
            raise ValueError("Paired runs use different tasks or ordering")
