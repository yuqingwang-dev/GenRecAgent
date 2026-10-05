"""Bounded, deterministic routing for offline history-based recommendations.

This is not a natural-language policy or a claim that generative retrieval beats Dense.
"""

from dataclasses import asdict, dataclass
from math import isfinite
from time import perf_counter
from typing import Literal

from genrecagent.retrieval.dense import DenseRetriever, SearchHit, mean_user_embedding
from genrecagent.retrieval.generative import GenerativeRetriever
from genrecagent.retrieval.popularity import PopularityRetriever

RouteName = Literal["dense", "generative", "popularity"]
RequestedRoute = Literal["auto", "dense", "generative", "popularity"]


@dataclass(frozen=True, slots=True)
class OfflineRouteRequest:
    history_items: tuple[str, ...]
    top_k: int = 20
    mode: RequestedRoute = "auto"
    allowed_item_ids: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.mode not in {"auto", "dense", "generative", "popularity"}:
            raise ValueError("mode must be auto, dense, generative, or popularity")
        if self.allowed_item_ids is not None and not isinstance(self.allowed_item_ids, frozenset):
            raise ValueError("allowed_item_ids must be an immutable set")


@dataclass(frozen=True, slots=True)
class RouteAttempt:
    route: RouteName
    success: bool
    returned_items: int
    latency_ms: float
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RoutedRecommendation:
    requested_mode: RequestedRoute
    selected_route: RouteName | None
    decision_reason: str
    hits: tuple[SearchHit, ...]
    attempts: tuple[RouteAttempt, ...]
    fallback_used: bool
    satisfied: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_mode": self.requested_mode,
            "selected_route": self.selected_route,
            "decision_reason": self.decision_reason,
            "hits": [hit.to_dict() for hit in self.hits],
            "attempts": [asdict(attempt) for attempt in self.attempts],
            "fallback_used": self.fallback_used,
            "satisfied": self.satisfied,
        }


class OfflineRecommendationRouter:
    """Dense-first route with explicit experimental generation and safe fallbacks."""

    def __init__(
        self,
        dense: DenseRetriever,
        popularity: PopularityRetriever,
        generative: GenerativeRetriever | None = None,
    ) -> None:
        if not dense.item_ids or set(popularity.ranked) != set(dense.item_ids):
            raise ValueError("Dense and popularity must share one nonempty item catalog")
        if generative is not None and set(generative.codebook.assignments) != set(dense.item_ids):
            raise ValueError("Generative and dense item catalogs must match")
        self.dense = dense
        self.popularity = popularity
        self.generative = generative
        self.catalog = frozenset(dense.item_ids)

    def route(self, request: OfflineRouteRequest) -> RoutedRecommendation:
        seen = set(request.history_items)
        allowed = self.catalog if request.allowed_item_ids is None else request.allowed_item_ids
        if request.allowed_item_ids is not None and not allowed.issubset(self.catalog):
            raise ValueError("Allowed item IDs must belong to the routing catalog")
        eligible_count = len(allowed) - sum(item_id in allowed for item_id in seen)
        required = min(request.top_k, eligible_count)
        if required == 0:
            return RoutedRecommendation(
                request.mode, None, "no_eligible_catalog_items", (), (), True, False
            )
        has_known_history = any(item_id in self.catalog for item_id in request.history_items)
        if request.mode == "popularity":
            routes: tuple[RouteName, ...] = ("popularity",)
            decision_reason = "explicit_popularity_route"
        elif request.mode == "generative":
            routes = (
                ("generative", "dense", "popularity") if has_known_history
                else ("generative", "popularity")
            )
            decision_reason = "explicit_experimental_generative_route"
        elif has_known_history:
            routes = ("dense", "popularity")
            decision_reason = "dense_first_measured_baseline"
        else:
            routes = ("popularity",)
            decision_reason = "cold_start_no_known_history"
        attempts: list[RouteAttempt] = []
        for route in routes:
            started = perf_counter()
            hits: tuple[SearchHit, ...] = ()
            reason: str | None = None
            try:
                if route == "dense":
                    user_vector = mean_user_embedding(request.history_items, self.dense)
                    if user_vector is None:
                        raise ValueError("No indexed item in history")
                    hits = tuple(self.dense.search(
                        user_vector, top_k=required, exclude_seen_items=seen,
                        allowed_item_ids=request.allowed_item_ids,
                    ))
                elif route == "generative":
                    if self.generative is None:
                        raise RuntimeError("Generative route is not configured")
                    result = self.generative.retrieve(request.history_items, top_k=required)
                    hits = result.hits
                    if any((
                        result.diagnostics.invalid_codes,
                        result.diagnostics.duplicate_codes,
                        result.diagnostics.duplicate_items,
                        result.diagnostics.candidate_shortfall,
                    )):
                        reason = "invalid_or_insufficient_generative_candidates"
                    reason = reason or self._invalid_hits(hits, route, seen, len(hits))
                    if request.allowed_item_ids is not None:
                        hits = tuple(
                            SearchHit(hit.item_id, hit.score, rank, hit.source)
                            for rank, hit in enumerate(
                                (hit for hit in hits if hit.item_id in allowed), start=1
                            )
                        )
                else:
                    hits = tuple(self.popularity.search(
                        top_k=required, exclude_seen_items=seen,
                        allowed_item_ids=request.allowed_item_ids,
                    ))
                reason = reason or self._invalid_hits(hits, route, seen, required)
                if reason is None and any(hit.item_id not in allowed for hit in hits):
                    reason = "ineligible_item"
            except Exception as exc:  # a bounded route failure must not crash fallback
                reason = f"{type(exc).__name__}: {exc}"
            attempts.append(RouteAttempt(
                route=route,
                success=reason is None,
                returned_items=len(hits),
                latency_ms=(perf_counter() - started) * 1000,
                reason=reason,
            ))
            if reason is None:
                return RoutedRecommendation(
                    requested_mode=request.mode,
                    selected_route=route,
                    decision_reason=decision_reason,
                    hits=hits,
                    attempts=tuple(attempts),
                    fallback_used=(route != routes[0] or (
                        route == "popularity" and request.mode in {"auto", "dense"}
                    )),
                    satisfied=True,
                )
        return RoutedRecommendation(
            request.mode, None, decision_reason, (), tuple(attempts), True, False
        )

    def _invalid_hits(
        self, hits: tuple[SearchHit, ...], route: RouteName,
        seen: set[str], required: int,
    ) -> str | None:
        if len(hits) != required:
            return "candidate_shortfall"
        ids = [hit.item_id for hit in hits]
        if len(set(ids)) != len(ids):
            return "duplicate_items"
        if any(item_id not in self.catalog or item_id in seen for item_id in ids):
            return "unknown_or_seen_item"
        if any(hit.rank != index or hit.source != route or not isfinite(hit.score)
               for index, hit in enumerate(hits, start=1)):
            return "invalid_rank_source_or_score"
        return None
