"""Deterministic Phase 0 planner with a future LLM seam."""

from collections.abc import Sequence

from genrecagent.agent.state import AgentState, UserRequest


class Planner:
    """Create a small baseline plan that exercises all core agent stages."""

    def plan(self, request: UserRequest) -> list[str]:
        del request
        return ["filter", "retrieval", "ranking"]

    def replan(self, request: UserRequest, state: AgentState) -> list[str]:
        """Return a bounded fallback plan after verification failure."""

        del request
        completed = {result.tool_name for result in state.tool_results}
        fallback = [step for step in ("filter", "retrieval", "ranking") if step not in completed]
        return fallback or ["retrieval", "ranking"]

    @staticmethod
    def explain(plan: Sequence[str]) -> str:
        return " -> ".join(plan)

