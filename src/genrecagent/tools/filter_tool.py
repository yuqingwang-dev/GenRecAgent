"""Deterministic constraint extraction and filtering tool."""

import re

from genrecagent.agent.state import AgentState, ToolResult, UserRequest
from genrecagent.tools.base import BaseTool


class FilterTool(BaseTool):
    name = "filter"
    description = "Extract and normalize simple price constraints from a request."

    def run(self, request: UserRequest, state: AgentState) -> ToolResult:
        del state
        constraints = dict(request.constraints)
        match = re.search(
            r"(?:under|below|less than)\s+\$?(\d+(?:\.\d+)?)", request.query.lower()
        )
        if match and "max_price" not in constraints:
            constraints["max_price"] = float(match.group(1))
        request.constraints = constraints
        return ToolResult(tool_name=self.name, success=True, data={"constraints": constraints})
