"""Tool execution and result normalization."""

from collections.abc import Mapping
from time import perf_counter

from genrecagent.agent.state import AgentState, ToolResult, UserRequest
from genrecagent.tools.base import BaseTool


class Executor:
    """Execute registered tools and update the shared trace state."""

    def __init__(self, tools: Mapping[str, BaseTool]) -> None:
        self.tools = dict(tools)

    def execute(self, tool_name: str, request: UserRequest, state: AgentState) -> ToolResult:
        tool = self.tools.get(tool_name)
        if tool is None:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}",
            )
            state.tool_results.append(result)
            return result

        started = perf_counter()
        try:
            result = tool.run(request, state)
        except Exception as exc:  # pragma: no cover - defensive boundary
            result = ToolResult(tool_name=tool_name, success=False, error=str(exc))
        result.latency_ms = (perf_counter() - started) * 1000
        state.tool_results.append(result)
        if isinstance(result.data, dict) and isinstance(result.data.get("items"), list):
            state.candidate_items = list(result.data["items"])
        return result

