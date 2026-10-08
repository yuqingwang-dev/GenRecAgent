import pytest

from genrecagent.agent.constraints import PriceLimit
from genrecagent.agent.tool_environment import (
    FixedStepPolicy,
    ToolEnvironment,
    ToolTask,
    run_tool_agent,
)
from test_tool_environment import _resources


@pytest.mark.parametrize("name", [[], {}, None, 123, True])
def test_malformed_name_is_a_recoverable_rejection_not_handler_crash(name):
    resources, items = _resources()
    env = ToolEnvironment(
        resources,
        items,
        ToolTask("bad-name", "recommend", ("A",), mode="dense", top_k=2, price_limit=PriceLimit(5)),
    )
    env.step({"name": "read_history", "arguments": {}})
    observation = env.step({"name": name, "arguments": {}})
    assert not observation["ok"] and "ValueError" in observation["error"]
    assert not env.source_errors and not env.trace[-1].get("service_failure")
    result = run_tool_agent(env, FixedStepPolicy())
    assert result["status"] == "success" and result["steps"] <= 10


def test_trace_keeps_actual_arguments_when_caller_later_mutates_its_dictionary():
    resources, items = _resources()
    env = ToolEnvironment(
        resources, items, ToolTask("immutable-action", "recommend", ("A",), mode="dense", top_k=2)
    )
    env.step({"name": "read_history", "arguments": {}})
    call = {"name": "retrieve_dense", "arguments": {"top_k": 50}}
    env.step(call)
    call["name"], call["arguments"]["top_k"] = "finish", 1
    assert env.trace[-1]["action"] == {"name": "retrieve_dense", "arguments": {"top_k": 50}}
