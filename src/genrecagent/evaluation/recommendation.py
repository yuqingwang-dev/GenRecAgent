"""Standard ranking metrics with no external dependencies."""

from collections.abc import Iterable, Sequence
from math import log2


def recall_at_k(predicted: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(set(predicted[:k]) & relevant_set) / len(relevant_set)


def ndcg_at_k(predicted: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    dcg = sum(
        1.0 / log2(index + 2)
        for index, item_id in enumerate(predicted[:k])
        if item_id in relevant_set
    )
    ideal_length = min(len(relevant_set), k)
    idcg = sum(1.0 / log2(index + 2) for index in range(ideal_length))
    return dcg / idcg if idcg else 0.0
