"""Verification rules for the Phase 0 recommendation trace."""

from genrecagent.agent.constraints import PriceLimit
from genrecagent.agent.state import AgentState, VerificationResult


class Verifier:
    """Check basic output validity and declared user constraints."""

    def verify(self, state: AgentState) -> VerificationResult:
        reasons: list[str] = []
        if not state.candidate_items:
            reasons.append("No candidate items were produced.")

        max_price = state.original_request.constraints.get("max_price")
        if max_price is not None:
            try:
                limit = PriceLimit(max_price)
                if any(not limit.allows(item.get("price")) for item in state.candidate_items):
                    reasons.append("At least one candidate has an unknown or over-budget price.")
            except ValueError:
                reasons.append("Invalid max_price constraint.")

        passed = not reasons
        return VerificationResult(
            passed=passed,
            reasons=tuple(reasons) if reasons else ("Candidates satisfy basic checks.",),
            need_replan=not passed,
        )
