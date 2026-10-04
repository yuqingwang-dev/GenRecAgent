"""Fuse two valid-code rankings without comparing their uncalibrated scores."""

from genrecagent.retrieval.semantic_trie import CodePrediction, SemanticID


def fuse_code_ranks(
    transition: tuple[CodePrediction, ...],
    transformer: tuple[CodePrediction, ...],
    *,
    transformer_weight: float,
    max_codes: int,
    rank_offset: int = 20,
) -> tuple[CodePrediction, ...]:
    """Weighted reciprocal ranks; exact endpoints retain the original order."""

    if not 0 <= transformer_weight <= 1:
        raise ValueError("transformer_weight must lie in [0, 1]")
    if max_codes < 1 or rank_offset < 0:
        raise ValueError("max_codes must be positive and rank_offset nonnegative")
    if transformer_weight == 0:
        return tuple(transition[:max_codes])
    if transformer_weight == 1:
        return tuple(transformer[:max_codes])
    combined: dict[SemanticID, float] = {}
    for predictions, weight in (
        (transition, 1 - transformer_weight),
        (transformer, transformer_weight),
    ):
        seen: set[SemanticID] = set()
        for prediction in predictions:
            if prediction.rank < 1 or prediction.semantic_id in seen:
                raise ValueError("Input predictions require unique codes and positive ranks")
            seen.add(prediction.semantic_id)
            combined[prediction.semantic_id] = combined.get(prediction.semantic_id, 0.0) + (
                weight / (rank_offset + prediction.rank)
            )
    ordered = sorted(combined.items(), key=lambda row: (-row[1], row[0]))[:max_codes]
    return tuple(
        CodePrediction(semantic_id=code, score=score, rank=rank)
        for rank, (code, score) in enumerate(ordered, start=1)
    )
