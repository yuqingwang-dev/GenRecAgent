"""Catalog-valid code decoding and target-independent candidate expansion.

No generative model is trained here. A future model implements NextTokenScorer.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from genrecagent.data.models import UserHistory
from genrecagent.retrieval.dense import SearchHit
from genrecagent.retrieval.semantic_trie import (
    CodePrediction,
    NextTokenScorer,
    SemanticCodeTrie,
    SemanticID,
)
from genrecagent.semantic_id.codebook import SemanticCodebook


@dataclass(frozen=True, slots=True)
class CandidateDiagnostics:
    predicted_codes: int
    collision_codes: int
    invalid_codes: int
    duplicate_codes: int
    expanded_items: int
    excluded_seen: int
    duplicate_items: int
    candidate_shortfall: int


@dataclass(frozen=True, slots=True)
class GenerativeSearchResult:
    hits: tuple[SearchHit, ...]
    predicted_codes: tuple[CodePrediction, ...]
    hit_codes: tuple[SemanticID, ...]
    diagnostics: CandidateDiagnostics

    def to_dict(self) -> dict[str, object]:
        return {
            "hits": [hit.to_dict() for hit in self.hits],
            "predicted_codes": [asdict(code) for code in self.predicted_codes],
            "hit_codes": [list(code) for code in self.hit_codes],
            "diagnostics": asdict(self.diagnostics),
        }


def train_popularity_counts(histories: Sequence[UserHistory]) -> Counter[str]:
    """Never include validation or test interactions in collision tie-breaking."""

    return Counter(item_id for user in histories for item_id in user.train_items)


def expand_code_predictions(
    predictions: Sequence[CodePrediction],
    codebook: SemanticCodebook,
    *,
    train_popularity: Mapping[str, int],
    seen_items: set[str] | frozenset[str],
    top_k: int,
) -> GenerativeSearchResult:
    """Order by code rank, then train-only popularity and stable item ID."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    seen_codes: set[SemanticID] = set()
    selected_items: set[str] = set()
    ordered: list[tuple[str, float, SemanticID]] = []
    collision_codes = invalid_codes = duplicate_codes = 0
    expanded_items = excluded_seen = duplicate_items = 0
    for prediction in sorted(predictions, key=lambda row: row.rank):
        code = prediction.semantic_id
        if code in seen_codes:
            duplicate_codes += 1
            continue
        seen_codes.add(code)
        group = codebook.lookup(code)
        if not group:
            invalid_codes += 1
            continue
        collision_codes += len(group) > 1
        expanded_items += len(group)
        for item_id in sorted(group, key=lambda item: (-train_popularity.get(item, 0), item)):
            if item_id in seen_items:
                excluded_seen += 1
            elif item_id in selected_items:
                duplicate_items += 1
            else:
                selected_items.add(item_id)
                ordered.append((item_id, prediction.score, code))
    chosen = ordered[:top_k]
    hits = tuple(
        SearchHit(item_id=item_id, score=score, rank=rank, source="generative")
        for rank, (item_id, score, _) in enumerate(chosen, start=1)
    )
    return GenerativeSearchResult(
        hits=hits,
        predicted_codes=tuple(predictions),
        hit_codes=tuple(code for _, _, code in chosen),
        diagnostics=CandidateDiagnostics(
            predicted_codes=len(predictions),
            collision_codes=collision_codes,
            invalid_codes=invalid_codes,
            duplicate_codes=duplicate_codes,
            expanded_items=expanded_items,
            excluded_seen=excluded_seen,
            duplicate_items=duplicate_items,
            candidate_shortfall=top_k - len(hits),
        ),
    )


class GenerativeRetriever:
    """Model-agnostic history-to-valid-code-to-item adapter."""

    def __init__(
        self,
        codebook: SemanticCodebook,
        scorer: NextTokenScorer,
        *,
        train_popularity: Mapping[str, int],
        beam_size: int = 20,
        max_codes: int = 20,
    ) -> None:
        if beam_size < 1 or max_codes < 1:
            raise ValueError("beam_size and max_codes must be positive")
        self.codebook = codebook
        self.scorer = scorer
        self.train_popularity = train_popularity
        self.beam_size = beam_size
        self.max_codes = max_codes
        self.trie = SemanticCodeTrie(codebook.assignments.values())

    def retrieve(self, history_items: Sequence[str], *, top_k: int = 20) -> GenerativeSearchResult:
        history = tuple(self.codebook.encode(item_id) for item_id in history_items)
        codes = self.trie.beam_search(
            self.scorer, history, beam_size=self.beam_size, max_codes=self.max_codes
        )
        return expand_code_predictions(
            codes,
            self.codebook,
            train_popularity=self.train_popularity,
            seen_items=set(history_items),
            top_k=top_k,
        )
