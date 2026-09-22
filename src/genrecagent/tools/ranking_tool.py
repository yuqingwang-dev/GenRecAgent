"""Deterministic ranking tool."""

from genrecagent.agent.state import AgentState, ToolResult, UserRequest
from genrecagent.tools.base import BaseTool


class RankingTool(BaseTool):
    name = "ranking"
    description = "Order the current candidates by their mock relevance score."

    def run(self, request: UserRequest, state: AgentState) -> ToolResult:
        del request
        items = sorted(state.candidate_items, key=lambda item: item.get("score", 0), reverse=True)
        return ToolResult(tool_name=self.name, success=True, data={"items": items})

