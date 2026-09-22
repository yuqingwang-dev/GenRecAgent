"""Verification rules for the Phase 0 recommendation trace."""

from genrecagent.agent.state import AgentState, VerificationResult


class Verifier:
    """Check basic output validity and declared user constraints."""

    def verify(self, state: AgentState) -> VerificationResult:
        reasons: list[str] = []
        if not state.candidate_items:
            reasons.append("No candidate items were produced.")

        max_price = state.original_request.constraints.get("max_price")
        if isinstance(max_price, (int, float)):
            over_limit = [
                item for item in state.candidate_items if float(item.get("price", 0)) > max_price
            ]
            if over_limit:
                reasons.append("At least one candidate violates max_price.")

        passed = not reasons
        return VerificationResult(
            passed=passed,
            reasons=tuple(reasons) if reasons else ("Candidates satisfy basic checks.",),
            need_replan=not passed,
        )

