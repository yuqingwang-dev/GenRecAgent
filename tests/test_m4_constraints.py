from dataclasses import replace

import numpy as np
import pytest

from genrecagent.agent.constrained_agent import ConstrainedRecommendationAgent
from genrecagent.agent.constraints import PriceLimit
from genrecagent.agent.intent import parse_recommendation_intent
from genrecagent.agent.offline_router import OfflineRecommendationRouter, OfflineRouteRequest
from genrecagent.agent.state import AgentState, UserRequest
from genrecagent.agent.verifier import Verifier
from genrecagent.data.models import Item, UserHistory
from genrecagent.evaluation.constraint_tasks import grade_constraint_task
from genrecagent.retrieval.dense import DenseRetriever, SearchHit
from genrecagent.retrieval.popularity import PopularityRetriever


def _agent() -> ConstrainedRecommendationAgent:
    dense = DenseRetriever()
    dense.build_index(
        ["A", "B", "C", "D", "E"],
        np.array([[1, 0], [1, .1], [.9, .2], [.8, .3], [0, 1]], dtype=np.float32),
    )
    history = UserHistory("u", ("A", "B"), "C", "D", (1, 2), 3, 4)
    popularity = PopularityRetriever([history], dense.item_ids)
    items = {
        item_id: Item(item_id, item_id, "", (), ("All Beauty",), price, None)
        for item_id, price in zip(dense.item_ids, (10., None, 50., 5., 10.), strict=True)
    }
    return ConstrainedRecommendationAgent(OfflineRecommendationRouter(dense, popularity), items)


@pytest.mark.parametrize("price", [None, True, "5", float("nan"), float("inf"), -1])
def test_unknown_or_invalid_price_does_not_pass_budget(price) -> None:
    assert not PriceLimit(10).allows(price)
    with pytest.raises(ValueError):
        PriceLimit(price)


def test_price_boundaries_and_controlled_intents() -> None:
    assert PriceLimit(10).allows(10)
    assert not PriceLimit(10, inclusive=False).allows(10)
    parsed = parse_recommendation_intent("Recommend based on my history under $10.")
    assert parsed.status == "accepted" and parsed.intent == "personalized"
    assert parsed.price_limit == PriceLimit(10, inclusive=False)
    parsed = parse_recommendation_intent("推荐热门商品，价格不超过20美元。")
    assert parsed.intent == "popular" and parsed.price_limit == PriceLimit(20)
    assert parse_recommendation_intent("Use generative retrieval.").intent == "generative"


@pytest.mark.parametrize("text", [
    "Recommend based on my history under -5 dollars.",
    "Recommend based on my history under NaN dollars.",
    "Recommend based on my history under $10 and at most $20.",
])
def test_malformed_or_conflicting_budget_is_rejected(text) -> None:
    assert parse_recommendation_intent(text).status == "invalid_request"


def test_unsupported_conditions_are_not_silently_discarded() -> None:
    assert parse_recommendation_intent(
        "根据我的历史推荐商品，价格不超过20元。"
    ).status == "unsupported_request"
    assert parse_recommendation_intent("Recommend moisturizer under $20.").status == (
        "unsupported_request"
    )


def test_prefilter_fills_budgeted_request_beyond_unconstrained_top_k() -> None:
    agent = _agent()
    request = "Recommend based on my history at most $10."
    result = agent.run(request, ("A",), top_k=2)
    assert result.status == "fulfilled" and result.constraints_verified
    assert [hit.item_id for hit in result.hits] == ["D", "E"]
    postfilter = agent.run(request, ("A",), top_k=2, prefilter=False)
    assert postfilter.status == "insufficient_candidates" and postfilter.eligible_count == 2
    assert postfilter.hits == ()


def test_insufficient_budget_returns_safe_partial_or_empty_without_relaxing() -> None:
    agent = _agent()
    result = agent.run("Recommend from my history under $10.", ("A",), top_k=2)
    assert result.status == "insufficient_candidates" and result.eligible_count == 1
    assert [hit.item_id for hit in result.hits] == ["D"]
    empty = agent.run("Show popular products at most $0.", top_k=2)
    assert empty.status == "insufficient_candidates" and empty.hits == ()
    assert empty.route_trace is None


def test_popularity_fallback_keeps_price_eligibility(monkeypatch) -> None:
    agent = _agent()

    def broken_search(*args, **kwargs):
        raise RuntimeError("simulated index failure")

    monkeypatch.setattr(agent.router.dense, "search", broken_search)
    result = agent.run("Recommend from my history at most $10.", ("A",), top_k=2)
    assert result.status == "fulfilled" and result.route_trace.selected_route == "popularity"
    assert {hit.item_id for hit in result.hits} == {"D", "E"}
    assert result.route_trace.fallback_used and all(price <= 10 for price in result.prices)


def test_explicit_popularity_and_invalid_count() -> None:
    agent = _agent()
    result = agent.run("Show popular products.", ("A",), top_k=2)
    assert result.route_trace.selected_route == "popularity"
    assert not result.route_trace.fallback_used
    invalid = agent.run("Show popular products.", top_k="invalid")
    assert invalid.status == "invalid_request" and invalid.to_dict()["items"] == []
    with pytest.raises(ValueError, match="catalog"):
        agent.router.route(OfflineRouteRequest(("A",), allowed_item_ids=frozenset({"BAD"})))


def test_phase_zero_verifier_rejects_missing_price_in_budgeted_request() -> None:
    request = UserRequest("recommend", constraints={"max_price": 10})
    state = AgentState(request, candidate_items=[{"item_id": "missing", "price": None}])
    assert not Verifier().verify(state).passed


def test_independent_grader_checks_metadata_and_does_not_reward_empty_feasible_output() -> None:
    agent = _agent()
    task = {
        "text": "Recommend based on my history at most $10.",
        "history_items": ["A"], "top_k": 2,
        "expected": {
            "status": "accepted", "intent": "personalized",
            "price_limit": {"amount": 10, "inclusive": True},
        },
    }
    result = agent.run(task["text"], ("A",), top_k=2)
    assert grade_constraint_task(task, result, agent.items)["contract_pass"]
    tampered = replace(
        result, hits=(SearchHit("B", 1.0, 1, "dense"),), prices=(0.0,),
    )
    grade = grade_constraint_task(task, tampered, agent.items)
    assert not grade["contract_pass"] and grade["unknown_price_admitted"] == 1
    empty = replace(result, hits=(), prices=(), status="insufficient_candidates")
    assert not grade_constraint_task(task, empty, agent.items)["contract_pass"]
