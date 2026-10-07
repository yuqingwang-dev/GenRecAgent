from types import SimpleNamespace

import pytest

from genrecagent.agent.constraints import PriceLimit
from genrecagent.agent.tool_environment import (
    FixedStepPolicy,
    ToolEnvironment,
    ToolTask,
    run_tool_agent,
)
from genrecagent.data.models import Item
from genrecagent.evaluation.agent_contracts import (
    end_reason,
    grade_contract,
    legal_action_attempt,
    summarize_contracts,
)
from genrecagent.evaluation.source_ablation import SourceAblationEnvironment
from test_tool_environment import _resources


def _task(expected="success"):
    return {
        "expected": expected,
        "max_price": 5,
        "top_k": 1,
        "category": None,
        "history": [],
        "scenario": "known",
    }


def _result(status, rows, trace=None):
    return {"status": status, "items": rows, "trace": trace or [], "steps": len(trace or [])}


def test_empty_result_has_no_conditional_constraint_rate_or_false_quantity_success():
    items = {"A": SimpleNamespace(price=2, categories=())}
    result = _result("success", [])
    grade = grade_contract(result, _task(), items)
    assert not grade["contract_pass"] and grade["nonempty_output_safe"] is None
    metrics = summarize_contracts([{"task": _task(), "result": result, "grade": grade}])
    assert metrics["hard_constraint_satisfaction_rate_on_nonempty"] is None
    assert metrics["feasible_fulfillment_rate"] == 0


def test_rank_score_and_audited_pool_are_checked_independently():
    items = {
        "A": SimpleNamespace(price=2, categories=()),
        "B": SimpleNamespace(price=2, categories=()),
    }
    row = {"item_id": "A", "rank": 1, "score": 1.0}
    audit = {
        "observation": {"ok": True},
        "candidate_audit": {"kind": "filtered_pools", "sources": {"dense": [{"item_id": "A"}]}},
    }
    good = grade_contract(_result("success", [row], [audit]), _task(), items)
    assert good["contract_pass"] and good["provenance_checked"]
    forged = grade_contract(_result("success", [{**row, "item_id": "B"}], [audit]), _task(), items)
    assert not forged["contract_pass"] and forged["provenance_checked"] is False
    for change in ({"rank": 0}, {"score": float("nan")}):
        assert not grade_contract(_result("success", [{**row, **change}]), _task(), items)[
            "contract_pass"
        ]


def test_service_failure_is_legal_and_external_seed_is_excluded_from_denominator():
    seeded = {"action": {"name": "finish"}, "observation": {"ok": False}}
    service = {
        "action": {"name": "retrieve_dense"},
        "observation": {"ok": False, "error": "TimeoutError"},
        "service_failure": True,
    }
    invalid = {"action": {"name": "filter_candidates"}, "observation": {"ok": False}}
    assert legal_action_attempt(service) and not legal_action_attempt(invalid)
    task = {**_task("clarification"), "scenario": "early_finish"}
    result = _result("clarification", [], [seeded, service, invalid])
    metrics = summarize_contracts(
        [
            {
                "task": task,
                "result": result,
                "grade": {"contract_pass": True, "nonempty_output_safe": None},
            }
        ]
    )
    assert metrics["agent_decision_attempts"] == 2
    assert metrics["legal_action_attempt_rate"] == 0.5
    assert metrics["tool_call_attempts"] == 2
    assert metrics["correct_no_result_rate"] == 1


def test_full_pool_audit_does_not_expand_model_visible_preview_and_timeout_recovers():
    resources, items = _resources()
    resources.dual.dense.build_index(
        [*items, "E", "F", "G"],
        [[1, 0], [0.9, 0.1], [0.5, 0.5], [0, 1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
    )
    for item in ("E", "F", "G"):
        items[item] = Item(item, item, "", (), ("All Beauty",), 3, None)
    env = ToolEnvironment(
        resources,
        items,
        ToolTask("audit", "recommend", ("A",), mode="dense", top_k=2, price_limit=PriceLimit(5)),
    )
    env.step({"name": "read_history", "arguments": {}})
    observation = env.step({"name": "retrieve_dense", "arguments": {"top_k": 50}})
    assert len(observation["items"]) == 3
    assert len(env.trace[-1]["candidate_audit"]["hits"]) == 6
    assert len(env.snapshot()["last_observation"]["items"]) == 3

    resources, items = _resources()

    def timed_out(*args, **kwargs):
        raise TimeoutError("read-only test backend deadline")

    resources.dual.dense.search = timed_out
    env = ToolEnvironment(
        resources,
        items,
        ToolTask("timeout", "recommend", ("A",), top_k=2, price_limit=PriceLimit(5)),
    )
    result = run_tool_agent(env, FixedStepPolicy())
    assert result["status"] == "success"
    assert any(step.get("service_failure") for step in result["trace"])
    assert end_reason(result) == "verified_count_fulfilled"


@pytest.mark.parametrize("disabled", ["retrieve_dense", "retrieve_generative"])
def test_source_removal_keeps_original_route_and_records_real_recovery(disabled):
    resources, items = _resources()
    task = ToolTask(
        "remove-source", "recommend", ("A",), mode="hybrid", top_k=2, price_limit=PriceLimit(5)
    )
    env = SourceAblationEnvironment(resources, items, task, disabled_tool=disabled)
    result = run_tool_agent(env, FixedStepPolicy())
    assert env.task.mode == "hybrid" and result["status"] == "success"
    removed = [step for step in result["trace"] if step["action"]["name"] == disabled]
    assert len(removed) == 1 and removed[0]["service_failure"]
    assert "source_disabled_for_ablation" in removed[0]["observation"]["error"]
    assert legal_action_attempt(removed[0])
