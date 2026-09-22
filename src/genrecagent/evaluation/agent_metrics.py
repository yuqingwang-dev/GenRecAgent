"""Agent-level metrics derived from a typed run trace."""

from dataclasses import dataclass

from genrecagent.agent.state import AgentState


@dataclass(slots=True)
class AgentMetrics:
    task_success_rate: float
    constraint_satisfaction_rate: float
    tool_selection_accuracy: float
    invalid_tool_call_rate: float
    average_tool_calls: float
    average_agent_steps: float
    latency_ms: float


def summarize_agent_run(state: AgentState) -> AgentMetrics:
    tool_calls = len(state.tool_results)
    invalid_calls = sum(not result.success for result in state.tool_results)
    latency = sum(result.latency_ms for result in state.tool_results)
    passed = bool(state.verification_result and state.verification_result.passed)
    return AgentMetrics(
        task_success_rate=float(passed),
        constraint_satisfaction_rate=float(passed),
        tool_selection_accuracy=float(invalid_calls == 0),
        invalid_tool_call_rate=invalid_calls / tool_calls if tool_calls else 0.0,
        average_tool_calls=float(tool_calls),
        average_agent_steps=float(state.step_count),
        latency_ms=latency,
    )

