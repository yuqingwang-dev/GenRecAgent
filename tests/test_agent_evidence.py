from copy import deepcopy
from types import SimpleNamespace

import pytest

from genrecagent.evaluation.agent_contracts import grade_contract, summarize_contracts
from genrecagent.evaluation.agent_evidence import audit_run, check_paired_runs


def _fixture():
    task = {
        "task_id": "val1",
        "expected": "success",
        "max_price": 5,
        "top_k": 1,
        "category": None,
        "history": [],
        "scenario": "known",
    }
    hit = {"item_id": "A", "rank": 1, "score": 0.9, "source": "dense"}
    trace = [
        {
            "action": {"name": "retrieve_dense", "arguments": {"top_k": 50}},
            "observation": {"ok": True, "source": "dense", "count": 1, "items": [hit]},
            "candidate_audit": {"kind": "retrieved_pool", "source": "dense", "hits": [hit]},
        },
        {
            "action": {"name": "filter_candidates", "arguments": {}},
            "observation": {"ok": True, "source_counts": {"dense": 1}},
            "candidate_audit": {"kind": "filtered_pools", "sources": {"dense": [hit]}},
        },
        {"action": {"name": "finish", "arguments": {}}, "observation": {"ok": True}},
    ]
    result = {
        "task_id": "val1",
        "status": "success",
        "steps": 3,
        "invalid_calls": 0,
        "items": [hit],
        "trace": trace,
        "end_reason": "verified_count_fulfilled",
    }
    items = {
        "A": SimpleNamespace(price=2, categories=()),
        "B": SimpleNamespace(price=2, categories=()),
    }
    row = {
        "task": task,
        "result": result,
        "grade": grade_contract(result, task, items),
        "contract_pass": True,
    }
    metrics = {
        "tasks": 1,
        "task_ids": ["val1"],
        "contract_pass": 1,
        "contract_pass_rate": 1,
        "max_steps": 10,
        "max_seconds": 120,
        "split": "validation",
        "task_file_sha256": "tasks",
        "base_model_revision": "qwen",
        "recommendation_checkpoint_sha256": "m3",
        "policy_illegal_calls": 0,
        "scope": "validation: full task split",
        "mean_steps": 3,
        "invalid_calls": 0,
        "feasible_tasks": 1,
        "feasible_filled": 1,
        "correct_clarifications": 0,
        "constraint_violations": 0,
        "returned_items": 1,
        "outcomes": {"success": 1},
        "contract_metrics": summarize_contracts([row]),
    }
    return metrics, [row], [deepcopy(task)], items


def test_full_evidence_regrades_without_importing_model_runtime():
    metrics, rows, tasks, items = _fixture()
    report = audit_run(metrics, rows, tasks, items)
    assert report["candidate_audits"] == 2 and report["nonempty_provenance_verified"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "task",
        "pool",
        "preview",
        "rate",
        "grade",
        "reason",
        "missing_task",
        "count",
        "zero_decisions",
    ],
)
def test_mutated_evidence_is_rejected(mutation):
    metrics, rows, tasks, items = _fixture()
    if mutation == "task":
        rows[0]["task"]["expected"] = "clarification"
    elif mutation == "pool":
        rows[0]["result"]["trace"][1]["candidate_audit"]["sources"]["dense"] = [
            {"item_id": "B", "rank": 1, "score": 0.9, "source": "dense"}
        ]
    elif mutation == "preview":
        rows[0]["result"]["trace"][0]["observation"]["items"] = []
    elif mutation == "rate":
        metrics["contract_metrics"]["legal_action_attempt_rate"] = 0.5
    elif mutation == "grade":
        rows[0]["grade"]["provenance_checked"] = None
    elif mutation == "reason":
        rows[0]["result"]["end_reason"] = "controller_success"
    elif mutation == "missing_task":
        tasks.append({**tasks[0], "task_id": "omitted"})
    elif mutation == "count":
        metrics["returned_items"] = 2
    else:
        rows[0]["result"]["trace"] = []
        rows[0]["result"]["steps"] = 0
    with pytest.raises(ValueError):
        audit_run(metrics, rows, tasks, items)


def test_comparison_rejects_changed_tasks_and_model_budgets():
    metrics, rows, _, _ = _fixture()
    check_paired_runs([(metrics, rows), (deepcopy(metrics), deepcopy(rows))])
    changed = {**metrics, "max_steps": 20}
    with pytest.raises(ValueError, match="max_steps"):
        check_paired_runs([(metrics, rows), (changed, rows)])
    changed_rows = deepcopy(rows)
    changed_rows[0]["task"]["history"] = ["B"]
    with pytest.raises(ValueError, match="different tasks"):
        check_paired_runs([(metrics, rows), (metrics, changed_rows)])


def test_no_decisions_is_not_a_perfect_legal_action_rate():
    metrics, rows, tasks, items = _fixture()
    result = rows[0]["result"]
    result.update(
        status="budget_exhausted",
        items=[],
        steps=0,
        trace=[],
        end_reason="execution_deadline_exhausted",
    )
    rows[0].update(grade=grade_contract(result, tasks[0], items), contract_pass=False)
    metrics.update(
        contract_pass=0,
        contract_pass_rate=0,
        mean_steps=0,
        feasible_filled=0,
        returned_items=0,
        outcomes={"budget_exhausted": 1},
        contract_metrics=summarize_contracts(rows),
    )
    report = audit_run(metrics, rows, tasks, items)
    assert report["contract_metrics"]["legal_action_attempt_rate"] is None
