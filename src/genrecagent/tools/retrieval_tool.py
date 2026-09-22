"""Small in-memory retrieval tool used only by the local demo."""

from genrecagent.agent.state import AgentState, ToolResult, UserRequest
from genrecagent.tools.base import BaseTool


class RetrievalTool(BaseTool):
    name = "retrieval"
    description = "Retrieve deterministic mock candidates with dense-retrieval-shaped fields."

    catalog = (
        {"item_id": "shoe-001", "title": "Daily Runner", "price": 79.0, "score": 0.95},
        {"item_id": "shoe-002", "title": "Trail Motion", "price": 99.0, "score": 0.91},
        {"item_id": "shoe-003", "title": "City Walk", "price": 129.0, "score": 0.84},
    )

    def run(self, request: UserRequest, state: AgentState) -> ToolResult:
        del state
        max_price = request.constraints.get("max_price")
        items = [dict(item) for item in self.catalog]
        if isinstance(max_price, (int, float)):
            items = [item for item in items if item["price"] <= float(max_price)]
        return ToolResult(tool_name=self.name, success=True, data={"items": items})

