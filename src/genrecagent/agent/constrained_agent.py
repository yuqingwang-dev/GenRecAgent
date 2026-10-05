"""Controlled-language intent, catalog filtering and independently checked output."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from genrecagent.agent.constraints import PriceLimit
from genrecagent.agent.intent import ParsedIntent, parse_recommendation_intent
from genrecagent.agent.offline_router import (
    OfflineRecommendationRouter,
    OfflineRouteRequest,
    RoutedRecommendation,
)
from genrecagent.data.models import Item
from genrecagent.retrieval.dense import SearchHit

OutcomeStatus = Literal[
    "fulfilled", "insufficient_candidates", "invalid_request", "unsupported_request", "tool_failure"
]


@dataclass(frozen=True, slots=True)
class ConstrainedRecommendation:
    parsed: ParsedIntent
    status: OutcomeStatus
    hits: tuple[SearchHit, ...]
    prices: tuple[float | None, ...]
    requested_count: int
    eligible_count: int | None
    constraints_verified: bool
    route_trace: RoutedRecommendation | None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "parsed": self.parsed.to_dict(), "status": self.status,
            "items": [
                {**hit.to_dict(), "metadata_price": price}
                for hit, price in zip(self.hits, self.prices, strict=True)
            ],
            "requested_count": self.requested_count, "eligible_count": self.eligible_count,
            "candidate_shortfall": max(0, self.requested_count - len(self.hits)),
            "constraints_verified": self.constraints_verified,
            "route_trace": self.route_trace.to_dict() if self.route_trace else None,
            "reason": self.reason,
        }


class ConstrainedRecommendationAgent:
    """Use a fixed intent grammar and retain hard eligibility through every fallback."""

    def __init__(self, router: OfflineRecommendationRouter, items: dict[str, Item]) -> None:
        if set(items) != set(router.catalog):
            raise ValueError("Constraint metadata and routing catalog must match")
        self.router = router
        self.items = dict(items)
        self._eligibility_cache: dict[PriceLimit, frozenset[str]] = {}

    def eligible_items(self, limit: PriceLimit | None) -> frozenset[str]:
        if limit is None:
            return self.router.catalog
        if limit in self._eligibility_cache:
            return self._eligibility_cache[limit]
        allowed = frozenset(
            item_id for item_id, item in self.items.items() if limit.allows(item.price)
        )
        if len(self._eligibility_cache) >= 32:
            self._eligibility_cache.pop(next(iter(self._eligibility_cache)))
        self._eligibility_cache[limit] = allowed
        return allowed

    def run(
        self, text: str, history_items: tuple[str, ...] = (), *, top_k: int = 5,
        prefilter: bool = True,
    ) -> ConstrainedRecommendation:
        parsed = parse_recommendation_intent(text)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 100:
            parsed = ParsedIntent("invalid_request", reason="top_k_out_of_range")
        if parsed.status != "accepted":
            safe_count = top_k if (
                isinstance(top_k, int) and not isinstance(top_k, bool) and top_k >= 0
            ) else 0
            return ConstrainedRecommendation(
                parsed, parsed.status, (), (), safe_count,
                None, False, None, parsed.reason,
            )
        allowed = self.eligible_items(parsed.price_limit)
        seen = set(history_items)
        eligible_count = len(allowed) - sum(item_id in allowed for item_id in seen)
        if prefilter and eligible_count == 0:
            return ConstrainedRecommendation(
                parsed, "insufficient_candidates", (), (), top_k, 0, True, None,
                "no_item_with_verified_metadata_satisfies_request",
            )
        mode = {"personalized": "auto", "popular": "popularity", "generative": "generative"}[
            parsed.intent
        ]
        result = self.router.route(OfflineRouteRequest(
            history_items, top_k=top_k, mode=mode,
            allowed_item_ids=allowed if prefilter and parsed.price_limit else None,
        ))
        hits = tuple(
            SearchHit(hit.item_id, hit.score, rank, hit.source)
            for rank, hit in enumerate(
                (hit for hit in result.hits if hit.item_id in allowed), start=1
            )
        )
        invalid = (
            len({hit.item_id for hit in hits}) != len(hits)
            or any(
                hit.item_id not in self.items or hit.item_id in seen
                or hit.rank != rank or not isfinite(hit.score)
                or (parsed.price_limit is not None and not parsed.price_limit.allows(
                    self.items[hit.item_id].price
                ))
                for rank, hit in enumerate(hits, start=1)
            )
        )
        if invalid:
            return ConstrainedRecommendation(
                parsed, "tool_failure", (), (), top_k, eligible_count, False, result,
                "output_verification_failed",
            )
        status: OutcomeStatus = "fulfilled" if len(hits) == top_k else "insufficient_candidates"
        if not result.satisfied and eligible_count >= top_k:
            status = "tool_failure"
        return ConstrainedRecommendation(
            parsed, status, hits, tuple(self.items[hit.item_id].price for hit in hits),
            top_k, eligible_count, True, result,
            None if status == "fulfilled" else "requested_count_not_available_or_retrieved",
        )
