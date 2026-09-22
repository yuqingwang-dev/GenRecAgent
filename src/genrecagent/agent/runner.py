"""End-to-end deterministic Phase 0 agent runner."""

from genrecagent.agent.executor import Executor
from genrecagent.agent.planner import Planner
from genrecagent.agent.router import Router
from genrecagent.agent.state import AgentState, UserRequest
from genrecagent.agent.verifier import Verifier
from genrecagent.tools.filter_tool import FilterTool
from genrecagent.tools.ranking_tool import RankingTool
from genrecagent.tools.retrieval_tool import RetrievalTool


class MockRecommendationAgent:
    """Bounded Planner -> Router -> Tool -> Verifier loop."""

    def __init__(self, max_steps: int = 8) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        self.planner = Planner()
        self.router = Router()
        self.executor = Executor(
            {
                "filter": FilterTool(),
                "retrieval": RetrievalTool(),
                "ranking": RankingTool(),
            }
        )
        self.verifier = Verifier()

    def run(self, request: UserRequest) -> AgentState:
        state = AgentState(original_request=request)
        state.current_plan = self.planner.plan(request)
        replan_used = False
        plan_index = 0

        while state.step_count < self.max_steps and not state.finished:
            if plan_index >= len(state.current_plan):
                verification = self.verifier.verify(state)
                state.verification_result = verification
                if verification.passed or replan_used:
                    state.finished = True
                    state.final_response = self._final_response(state)
                    break
                state.current_plan = self.planner.replan(request, state)
                replan_used = True
                plan_index = 0
                continue

            action = state.current_plan[plan_index]
            plan_index += 1
            tool_name = self.router.route(action)
            state.selected_tool = tool_name
            state.step_count += 1
            self.executor.execute(tool_name or action, request, state)

        if not state.finished:
            state.verification_result = self.verifier.verify(state)
            state.finished = True
            state.final_response = self._final_response(state)
        return state

    @staticmethod
    def _final_response(state: AgentState) -> str:
        if state.verification_result and state.verification_result.passed:
            ids = [str(item.get("item_id")) for item in state.candidate_items]
            return f"Recommended {len(ids)} items: {', '.join(ids)}."
        reason = (
            "; ".join(state.verification_result.reasons)
            if state.verification_result
            else "Unknown error"
        )
        return f"Unable to produce a verified recommendation: {reason}"

