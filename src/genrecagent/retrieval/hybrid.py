"""Real item-level Dense + trained Semantic-ID retrieval with rank fusion."""

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

from genrecagent.retrieval.dense import DenseRetriever, SearchHit, mean_user_embedding
from genrecagent.retrieval.generative import GenerativeRetriever


def fuse_item_ranks(
    dense: tuple[SearchHit, ...], generative: tuple[SearchHit, ...], *,
    dense_weight: float = .75, top_k: int = 20, rank_offset: int = 20,
    seen: frozenset[str] = frozenset(), allowed: frozenset[str] | None = None,
) -> tuple[SearchHit, ...]:
    if not 0 <= dense_weight <= 1 or top_k < 1 or rank_offset < 0:
        raise ValueError("Invalid fusion settings")
    scores: dict[str, float] = {}
    for hits, weight in ((dense, dense_weight), (generative, 1 - dense_weight)):
        if weight == 0:
            continue
        observed = set()
        for hit in hits:
            if hit.rank < 1 or not isfinite(hit.score) or hit.item_id in observed:
                raise ValueError("A source requires unique items, positive ranks and finite scores")
            observed.add(hit.item_id)
            if hit.item_id in seen or (allowed is not None and hit.item_id not in allowed):
                continue
            scores[hit.item_id] = scores.get(hit.item_id, 0.) + weight / (rank_offset + hit.rank)
    ordered = sorted(scores, key=lambda item: (-scores[item], item))[:top_k]
    return tuple(SearchHit(item, scores[item], rank, "hybrid")
                 for rank, item in enumerate(ordered, start=1))


@dataclass(frozen=True, slots=True)
class DualCandidates:
    dense: tuple[SearchHit, ...]
    generative: tuple[SearchHit, ...]
    errors: dict[str, str]
    generative_diagnostics: dict[str, Any]

    def fuse(
        self, *, dense_weight: float = .75, top_k: int = 20,
        seen: frozenset[str] = frozenset(), allowed: frozenset[str] | None = None,
    ) -> tuple[SearchHit, ...]:
        return fuse_item_ranks(
            self.dense, self.generative, dense_weight=dense_weight, top_k=top_k,
            seen=seen, allowed=allowed,
        )


class DualRetriever:
    def __init__(self, dense: DenseRetriever, generative: GenerativeRetriever) -> None:
        if set(dense.item_ids) != set(generative.codebook.assignments):
            raise ValueError("Dense and trained generative catalogs must match")
        self.dense, self.generative = dense, generative

    def candidates(
        self, history: tuple[str, ...], *, source_k: int = 50,
        allowed: frozenset[str] | None = None,
    ) -> DualCandidates:
        if source_k < 1:
            raise ValueError("source_k must be positive")
        errors, diagnostics = {}, {}
        dense_hits: tuple[SearchHit, ...] = ()
        generated_hits: tuple[SearchHit, ...] = ()
        try:
            vector = mean_user_embedding(history, self.dense)
            if vector is not None:
                dense_hits = tuple(self.dense.search(
                    vector, top_k=source_k, exclude_seen_items=set(history),
                    allowed_item_ids=allowed,
                ))
        except Exception as exc:
            errors["dense"] = f"{type(exc).__name__}: {exc}"
        try:
            result = self.generative.retrieve(history, top_k=source_k)
            diagnostics = asdict(result.diagnostics)
            generated_hits = result.hits
            if any(diagnostics[key] for key in (
                "invalid_codes", "duplicate_codes", "duplicate_items"
            )):
                raise ValueError("Invalid generative source candidates")
        except Exception as exc:
            generated_hits = ()
            errors["generative"] = f"{type(exc).__name__}: {exc}"
        return DualCandidates(dense_hits, generated_hits, errors, diagnostics)
