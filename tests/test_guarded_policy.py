from copy import deepcopy

import pytest

from genrecagent.agent.constraints import PriceLimit
from genrecagent.agent.guarded_policy import allowed_actions, rejection_reason, run_guarded_agent
from genrecagent.agent.tool_environment import FixedStepPolicy, ToolEnvironment, ToolTask
from test_tool_environment import _resources


class RepeatingPolicy:
    def __init__(self, name="filter_candidates"):
        self.name = name
        self.calls = 0

    def choose(self, state):
        self.calls += 1
        return {
            "name": self.name,
            "arguments": {"reason": "stop"} if self.name == "clarify" else {},
        }


def _environment(**kwargs):
    resources, items = _resources()
    return ToolEnvironment(
        resources,
        items,
        ToolTask("guard-test", "recommend", ("A",), top_k=2, price_limit=PriceLimit(5), **kwargs),
    )


def test_state_gate_blocks_early_filter_finish_and_clarification():
    state = _environment().snapshot()
    assert allowed_actions(state) == ("read_history",)
    for name in ("filter_candidates", "finish", "clarify"):
        call = {"name": name, "arguments": {"reason": "empty"} if name == "clarify" else {}}
        assert rejection_reason(call, state, allowed_actions(state))
    state["history_read"] = True
    allowed = allowed_actions(state)
    assert set(allowed) == {"retrieve_dense", "retrieve_generative"}
    for count in (True, 1, 1.5, 101):
        assert rejection_reason(
            {"name": "retrieve_dense", "arguments": {"top_k": count}}, state, allowed
        )
    assert (
        rejection_reason({"name": "retrieve_dense", "arguments": {"top_k": 50}}, state, allowed)
        is None
    )
    assert rejection_reason(
        {"name": "read_history", "arguments": {"max_price": 100}}, state, allowed
    )


@pytest.mark.parametrize("name", ["filter_candidates", "finish", "clarify"])
def test_rejected_proposals_handoff_is_explicit_and_does_not_spend_tool_budget(name):
    policy = RepeatingPolicy(name)
    result = run_guarded_agent(_environment(), policy)
    assert result["status"] == "success" and result["steps"] <= 10
    assert result["invalid_calls"] == 0
    assert policy.calls == 2
    assert result["guard"]["model_rejected_proposals"] == 2
    assert result["guard"]["handoff_to_controller"]
    assert result["guard"]["model_accepted_actions"] == 0
    assert result["guard"]["controller_actions"] == result["steps"]
    assert all(row["guard_decision"]["source"] == "controller" for row in result["trace"])
    assert all(row["metadata_price"] <= 5 for row in result["items"])


def test_conforming_model_actions_are_preserved_and_source_fault_can_recover():
    result = run_guarded_agent(
        _environment(injected_failures=frozenset({"retrieve_dense"})), FixedStepPolicy()
    )
    assert result["status"] == "success"
    assert result["invalid_calls"] == 1  # the seeded physical source fault, not a model rejection
    assert result["guard"]["controller_actions"] == 0
    assert result["guard"]["model_rejected_proposals"] == 0
    assert not result["guard"]["handoff_to_controller"]


def test_infeasible_or_rejected_request_uses_environment_rule_without_model_calls():
    for kwargs in ({"category": "Missing Category"}, {"request_status": "unsupported_request"}):
        policy = RepeatingPolicy("finish")
        result = run_guarded_agent(_environment(**kwargs), policy)
        assert result["status"] == "clarification" and not result["items"]
        assert policy.calls == 0 and result["guard"]["controller_actions"] == 1


def test_popularity_failure_is_not_retried_and_deadline_does_not_call_model():
    state = _environment().snapshot()
    state.update(
        history_read=True,
        pending_sources=[],
        source_counts={"dense": 0},
        source_errors={"popularity": "physical_failure"},
        filtered=True,
        ranked_ready=True,
        ranked_count=0,
        verified=True,
        sufficient=False,
    )
    before = deepcopy(state)
    assert allowed_actions(state) == ("clarify",)
    assert state == before
    env = _environment()
    env._started -= 1000
    policy = RepeatingPolicy()
    result = run_guarded_agent(env, policy)
    assert result["status"] == "budget_exhausted" and policy.calls == 0
