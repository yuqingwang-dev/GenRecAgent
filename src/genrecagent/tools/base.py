"""Base interface for tools that can be routed by the agent."""

from abc import ABC, abstractmethod

from genrecagent.agent.state import AgentState, ToolResult, UserRequest


class BaseTool(ABC):
    """Small stable contract for future retrieval and policy tools."""

    name: str
    description: str

    @abstractmethod
    def run(self, request: UserRequest, state: AgentState) -> ToolResult:
        """Execute the tool for one request."""

