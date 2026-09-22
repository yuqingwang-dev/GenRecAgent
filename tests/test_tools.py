from genrecagent.agent.runner import MockRecommendationAgent
from genrecagent.agent.state import AgentState, UserRequest
from genrecagent.evaluation.recommendation import ndcg_at_k, recall_at_k
from genrecagent.tools.filter_tool import FilterTool


def test_filter_tool_extracts_price_constraint() -> None:
    request = UserRequest(query="shoes under 100 dollars")
    result = FilterTool().run(request, AgentState(original_request=request))

    assert result.success is True
    assert result.data["constraints"]["max_price"] == 100.0
    assert request.constraints["max_price"] == 100.0


def test_mock_pipeline_finishes_with_verified_candidates() -> None:
    result = MockRecommendationAgent().run(
        UserRequest(query="running shoes", constraints={"max_price": 100.0})
    )

    assert result.finished is True
    assert result.verification_result is not None
    assert result.verification_result.passed is True
    assert [item["item_id"] for item in result.candidate_items] == ["shoe-001", "shoe-002"]
    assert result.step_count <= 8


def test_recommendation_metrics() -> None:
    predicted = ["a", "b", "c"]
    assert recall_at_k(predicted, {"a", "c"}, 2) == 0.5
    assert 0.0 < ndcg_at_k(predicted, {"a", "c"}, 3) <= 1.0
