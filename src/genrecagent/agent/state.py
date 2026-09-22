"""Typed state objects shared by the planner, tools, and verifier."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class UserRequest:
    """A user request entering the recommendation agent."""

    query: str
    user_id: str = "anonymous"
    user_history: tuple[str, ...] = ()
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolResult:
    """A normalized result returned by every tool execution."""

    tool_name: str
    success: bool
    data: Any = None
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VerificationResult:
    """Verifier output used to finish or re-plan a run."""

    passed: bool
    reasons: tuple[str, ...] = ()
    need_replan: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentState:
    """Complete, serializable trace state for one agent request."""

    original_request: UserRequest
    current_plan: list[str] = field(default_factory=list)
    selected_tool: str | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    candidate_items: list[dict[str, Any]] = field(default_factory=list)
    verification_result: VerificationResult | None = None
    step_count: int = 0
    finished: bool = False
    final_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_request": self.original_request.to_dict(),
            "current_plan": list(self.current_plan),
            "selected_tool": self.selected_tool,
            "tool_results": [result.to_dict() for result in self.tool_results],
            "candidate_items": list(self.candidate_items),
            "verification_result": self.verification_result.to_dict()
            if self.verification_result
            else None,
            "step_count": self.step_count,
            "finished": self.finished,
            "final_response": self.final_response,
        }

