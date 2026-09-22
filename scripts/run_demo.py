"""Run one deterministic end-to-end mock agent trace."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.agent.runner import MockRecommendationAgent
from genrecagent.agent.state import UserRequest


def main() -> None:
    request = UserRequest(
        query="running shoes under 100 dollars",
        user_id="demo-user",
        user_history=("running", "fitness"),
        constraints={"max_price": 100.0},
    )
    result = MockRecommendationAgent().run(request)

    print("Query")
    print(f"  {request.query}")
    print("Planner")
    print(f"  {' -> '.join(result.current_plan)}")
    for tool_result in result.tool_results:
        print(f"{tool_result.tool_name}")
        print(f"  success={tool_result.success}, latency_ms={tool_result.latency_ms:.3f}")
    print("Verifier")
    print(f"  passed={result.verification_result.passed if result.verification_result else False}")
    print("Final Result")
    print(f"  {result.final_response}")


if __name__ == "__main__":
    main()
