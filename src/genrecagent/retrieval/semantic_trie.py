"""Catalog-valid Semantic ID decoding independent of a model framework."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

SemanticID = tuple[str, ...]
HistoryCodes = tuple[SemanticID, ...]


class NextTokenScorer(Protocol):
    """Future Transformer adapter supplies log-probabilities over the token vocabulary."""

    def log_probabilities(
        self, history: HistoryCodes, prefix: SemanticID
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True, slots=True)
class CodePrediction:
    semantic_id: SemanticID
    score: float
    rank: int


class SemanticCodeTrie:
    """Restrict each generated token to prefixes of actual catalog codes."""

    def __init__(self, codes: Iterable[SemanticID]) -> None:
        unique = set(codes)
        if not unique:
            raise ValueError("A catalog with at least one Semantic ID is required")
        lengths = {len(code) for code in unique}
        if len(lengths) != 1 or 0 in lengths or any(not all(code) for code in unique):
            raise ValueError("Catalog Semantic IDs must have one nonempty fixed length")
        self.depth = lengths.pop()
        self.codes = frozenset(unique)
        transitions: dict[SemanticID, set[str]] = {}
        for code in unique:
            for level, token in enumerate(code):
                transitions.setdefault(code[:level], set()).add(token)
        self.transitions = {
            prefix: tuple(sorted(tokens)) for prefix, tokens in transitions.items()
        }

    def allowed_next(self, prefix: SemanticID) -> tuple[str, ...]:
        return self.transitions.get(prefix, ())

    def is_valid(self, code: SemanticID) -> bool:
        return code in self.codes

    def beam_search(
        self,
        scorer: NextTokenScorer,
        history: HistoryCodes,
        *,
        beam_size: int,
        max_codes: int,
    ) -> tuple[CodePrediction, ...]:
        if beam_size < 1 or max_codes < 1:
            raise ValueError("beam_size and max_codes must be positive")
        beam: list[tuple[SemanticID, float]] = [((), 0.0)]
        for _ in range(self.depth):
            expansions: list[tuple[SemanticID, float]] = []
            for prefix, cumulative in beam:
                log_probs = scorer.log_probabilities(history, prefix)
                for token in self.allowed_next(prefix):
                    score = log_probs.get(token, -math.inf)
                    if math.isfinite(score):
                        expansions.append(((*prefix, token), cumulative + score))
            if not expansions:
                return ()
            expansions.sort(key=lambda row: (-row[1], row[0]))
            beam = expansions[:beam_size]
        valid = [(code, score) for code, score in beam if self.is_valid(code)]
        return tuple(
            CodePrediction(semantic_id=code, score=score, rank=rank)
            for rank, (code, score) in enumerate(valid[:max_codes], start=1)
        )
