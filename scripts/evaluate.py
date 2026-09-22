"""Run the local Phase 0 smoke evaluation, not a benchmark experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.agent.runner import MockRecommendationAgent
from genrecagent.agent.state import UserRequest
from genrecagent.evaluation.agent_metrics import summarize_agent_run


def main() -> None:
    result = MockRecommendationAgent().run(
        UserRequest(query="running shoes under 100 dollars", user_id="eval-smoke")
    )
    metrics = summarize_agent_run(result)
    print(metrics)
    print("This is a software smoke check, not a reported benchmark result.")


if __name__ == "__main__":
    main()
