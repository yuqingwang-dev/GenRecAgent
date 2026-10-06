from types import SimpleNamespace

import numpy as np

from genrecagent.agent.constraints import PriceLimit
from genrecagent.agent.tool_environment import (
    FixedStepPolicy,
    QwenStepPolicy,
    ToolEnvironment,
    ToolTask,
    run_tool_agent,
)
from genrecagent.data.models import Item, UserHistory
from genrecagent.retrieval.dense import DenseRetriever, SearchHit
from genrecagent.retrieval.popularity import PopularityRetriever


def _resources():
    dense = DenseRetriever()
    dense.build_index(
        ["A", "B", "C", "D"], np.array([[1, 0], [0.9, 0.1], [0.5, 0.5], [0, 1]], dtype=np.float32)
    )
    popularity = PopularityRetriever(
        [UserHistory("u", ("A", "B"), "C", "D", (1, 2), 3, 4)], dense.item_ids
    )

    class Generator:
        def retrieve(self, history, *, top_k):
            return SimpleNamespace(
                hits=tuple(
                    SearchHit(item, -float(rank), rank, "generative")
                    for rank, item in enumerate((x for x in ("B", "C", "D") if x not in history), 1)
                )[:top_k]
            )

    resources = SimpleNamespace(
        dual=SimpleNamespace(dense=dense, generative=Generator()),
        base=SimpleNamespace(router=SimpleNamespace(popularity=popularity)),
    )
    items = {
        item: Item(item, item, "", (), ("All Beauty",), price, None)
        for item, price in zip(dense.item_ids, (1.0, None, 3.0, 4.0), strict=True)
    }
    return resources, items


def test_real_tools_filter_verify_and_stop_with_candidate_provenance() -> None:
    resources, items = _resources()
    env = ToolEnvironment(
        resources,
        items,
        ToolTask("normal", "recommend", ("A",), top_k=2, price_limit=PriceLimit(5)),
    )
    result = run_tool_agent(env, FixedStepPolicy())
    assert result["status"] == "success" and len(result["items"]) == 2
    assert {row["item_id"] for row in result["items"]} == {"C", "D"}
    names = [row["action"]["name"] for row in result["trace"]]
    assert names == [
        "read_history",
        "retrieve_dense",
        "retrieve_generative",
        "filter_candidates",
        "rank_candidates",
        "verify_candidates",
        "finish",
    ]


def test_tool_failure_is_observed_and_other_source_recovers() -> None:
    resources, items = _resources()
    env = ToolEnvironment(
        resources,
        items,
        ToolTask(
            "fault",
            "recommend",
            ("A",),
            top_k=2,
            injected_failures=frozenset({"retrieve_dense"}),
        ),
    )
    result = run_tool_agent(env, FixedStepPolicy())
    assert result["status"] == "success" and result["invalid_calls"] == 1
    assert "synthetic_injected_tool_failure" in result["trace"][1]["observation"]["error"]
    assert result["trace"][0]["state"]["source_errors"] == {}
    assert result["trace"][1]["state"]["source_errors"] == {}


def test_empty_price_or_category_never_relaxes_constraints() -> None:
    resources, items = _resources()
    for task in (
        ToolTask("zero", "recommend", (), price_limit=PriceLimit(0)),
        ToolTask("category", "recommend", (), category="Missing Category"),
    ):
        result = run_tool_agent(ToolEnvironment(resources, items, task), FixedStepPolicy())
        assert result["status"] == "clarification" and result["items"] == []


def test_early_finish_repetition_and_step_budget_are_enforced() -> None:
    resources, items = _resources()
    env = ToolEnvironment(resources, items, ToolTask("limits", "recommend", ("A",)), max_steps=2)
    first = env.step({"name": "finish", "arguments": {}})
    second = env.step({"name": "finish", "arguments": {}})
    assert not first["ok"] and "Verified" in first["error"]
    assert not second["ok"] and "Repeated" in second["error"]
    result = run_tool_agent(env, FixedStepPolicy())
    assert result["status"] == "budget_exhausted" and result["items"] == []


def test_expired_deadline_with_model_policy_terminates_without_a_trace_index_error() -> None:
    resources, items = _resources()
    llm = SimpleNamespace(tool_call=lambda *args: {"name": "read_history", "arguments": {}})
    env = ToolEnvironment(resources, items, ToolTask("expired", "recommend", ("A",)))
    env._started -= 1000
    result = run_tool_agent(env, QwenStepPolicy(llm))
    assert result["status"] == "budget_exhausted" and result["trace"] == []


def test_precondition_error_does_not_disable_a_source_after_state_is_repaired() -> None:
    resources, items = _resources()
    env = ToolEnvironment(resources, items, ToolTask("repair", "recommend", ("A",)))
    call = {"name": "retrieve_dense", "arguments": {"top_k": 5}}
    assert not env.step(call)["ok"]
    assert "dense" not in env.source_errors
    assert env.step({"name": "read_history", "arguments": {}})["ok"]
    assert env.step(call)["ok"]
    assert not env.step(call)["ok"]


def test_rejected_request_cannot_be_overridden_by_model_tool_actions() -> None:
    resources, items = _resources()
    env = ToolEnvironment(
        resources,
        items,
        ToolTask("rejected", "unsupported condition", ("A",), request_status="unsupported_request"),
    )
    assert not env.step({"name": "read_history", "arguments": {}})["ok"]
    assert not env.step({"name": "finish", "arguments": {}})["ok"]
    assert env.step({"name": "clarify", "arguments": {"reason": "unsupported"}})["ok"]
    assert env.result()["items"] == []
