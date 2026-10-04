import numpy as np
import pytest

from genrecagent.agent.offline_router import OfflineRecommendationRouter, OfflineRouteRequest
from genrecagent.data.models import UserHistory
from genrecagent.retrieval.dense import DenseRetriever, SearchHit
from genrecagent.retrieval.generative import CandidateDiagnostics, GenerativeSearchResult
from genrecagent.retrieval.popularity import PopularityRetriever
from genrecagent.semantic_id.codebook import SemanticCodebook


def _routes() -> tuple[DenseRetriever, PopularityRetriever]:
    dense = DenseRetriever()
    dense.build_index(
        ["A", "B", "C", "D"],
        np.array([[1, 0], [0.8, 0.2], [0, 1], [-1, 0]], dtype=np.float32),
    )
    history = UserHistory("u", ("A", "B"), "C", "D", (1, 2), 3, 4)
    return dense, PopularityRetriever([history], dense.item_ids)


class FakeGenerator:
    def __init__(self, hits: tuple[SearchHit, ...], shortfall: int = 0) -> None:
        self.codebook = SemanticCodebook(assignments={
            item: (f"<s0:{index}>", "<s1:0>", "<s2:0>")
            for index, item in enumerate(("A", "B", "C", "D"))
        })
        self.hits = hits
        self.shortfall = shortfall
        self.calls = 0

    def retrieve(self, history_items, *, top_k):
        del history_items, top_k
        self.calls += 1
        return GenerativeSearchResult(
            self.hits, (), (), CandidateDiagnostics(
                2, 0, 0, 0, len(self.hits), 0, 0, self.shortfall
            ),
        )


def test_auto_uses_dense_without_calling_weaker_generator() -> None:
    dense, popularity = _routes()
    generator = FakeGenerator(())
    router = OfflineRecommendationRouter(dense, popularity, generator)
    result = router.route(OfflineRouteRequest(("A",), top_k=2))
    expected = dense.search(
        np.array([1, 0], dtype=np.float32), top_k=2, exclude_seen_items={"A"}
    )
    assert [hit.item_id for hit in result.hits] == [hit.item_id for hit in expected]
    assert result.selected_route == "dense"
    assert result.satisfied and not result.fallback_used
    assert [attempt.route for attempt in result.attempts] == ["dense"]
    assert generator.calls == 0
    assert result.to_dict()["hits"][0]["source"] == "dense"


def test_cold_start_routes_directly_to_train_only_popularity() -> None:
    dense, popularity = _routes()
    router = OfflineRecommendationRouter(dense, popularity)
    for history in ((), ("UNKNOWN",)):
        result = router.route(OfflineRouteRequest(history, top_k=2))
        assert result.selected_route == "popularity"
        assert result.fallback_used and result.satisfied
        assert [hit.item_id for hit in result.hits] == ["A", "B"]
        assert [attempt.route for attempt in result.attempts] == ["popularity"]


def test_explicit_generative_route_and_invalid_candidate_fallback() -> None:
    dense, popularity = _routes()
    good = FakeGenerator((
        SearchHit("B", -1.0, 1, "generative"),
        SearchHit("C", -2.0, 2, "generative"),
    ))
    result = OfflineRecommendationRouter(dense, popularity, good).route(
        OfflineRouteRequest(("A",), top_k=2, mode="generative")
    )
    assert result.selected_route == "generative"
    assert not result.fallback_used and result.satisfied
    assert good.calls == 1

    bad = FakeGenerator((
        SearchHit("B", -1.0, 1, "generative"),
        SearchHit("B", -2.0, 2, "generative"),
    ))
    result = OfflineRecommendationRouter(dense, popularity, bad).route(
        OfflineRouteRequest(("A",), top_k=2, mode="generative")
    )
    assert result.selected_route == "dense" and result.fallback_used
    assert [attempt.route for attempt in result.attempts] == ["generative", "dense"]
    assert result.attempts[0].reason == "duplicate_items"
    assert all(hit.source == "dense" for hit in result.hits)


def test_shortfall_and_dense_exception_use_bounded_fallback(monkeypatch) -> None:
    dense, popularity = _routes()
    generator = FakeGenerator((SearchHit("B", -1.0, 1, "generative"),), shortfall=1)
    router = OfflineRecommendationRouter(dense, popularity, generator)
    result = router.route(OfflineRouteRequest(("A",), top_k=2, mode="generative"))
    assert result.selected_route == "dense"
    assert result.attempts[0].reason == "invalid_or_insufficient_generative_candidates"

    def fail_search(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(dense, "search", fail_search)
    result = router.route(OfflineRouteRequest(("A",), top_k=2))
    assert result.selected_route == "popularity"
    assert [attempt.route for attempt in result.attempts] == ["dense", "popularity"]
    assert "index unavailable" in result.attempts[0].reason
    assert result.satisfied and result.fallback_used


def test_no_eligible_items_and_invalid_requests_do_not_fake_success() -> None:
    dense, popularity = _routes()
    router = OfflineRecommendationRouter(dense, popularity)
    result = router.route(OfflineRouteRequest(("A", "B", "C", "D"), top_k=2))
    assert not result.satisfied and result.selected_route is None
    assert result.hits == ()
    with pytest.raises(ValueError, match="top_k"):
        OfflineRouteRequest(("A",), top_k=0)
    with pytest.raises(ValueError, match="mode"):
        OfflineRouteRequest(("A",), mode="not_a_route")


def test_catalog_mismatch_rejected_before_routing() -> None:
    dense, popularity = _routes()
    popularity.ranked.pop()
    with pytest.raises(ValueError, match="catalog"):
        OfflineRecommendationRouter(dense, popularity)
