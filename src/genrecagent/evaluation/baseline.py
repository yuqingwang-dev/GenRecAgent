"""Identical catalog and leave-last-out protocol for dense and popularity."""

from collections.abc import Sequence
from statistics import mean
from time import perf_counter
from typing import Any, Literal

from genrecagent.data.models import UserHistory
from genrecagent.evaluation.recommendation import ndcg_at_k, recall_at_k
from genrecagent.retrieval.dense import DenseRetriever, SearchHit, mean_user_embedding
from genrecagent.retrieval.popularity import PopularityRetriever


def evaluate_baselines(
    histories: Sequence[UserHistory],
    retriever: DenseRetriever,
    *,
    split: Literal["validation", "test"],
    k_values: Sequence[int] = (5, 10, 20),
) -> dict[str, Any]:
    if not k_values or min(k_values) <= 0:
        raise ValueError("k_values must be positive")
    item_set = set(retriever.item_ids)
    popularity = PopularityRetriever(list(histories), retriever.item_ids)
    accumulators: dict[str, dict[str, Any]] = {
        name: {"metrics": {f"Recall@{k}": [] for k in k_values},
               "ndcg": {f"NDCG@{k}": [] for k in k_values},
               "latency_ms": [], "no_history_embedding": 0}
        for name in ("popularity", "dense")
    }
    unreachable = 0
    target_in_history = 0
    max_k = max(k_values)
    for user in histories:
        history = user.validation_history if split == "validation" else user.test_history
        target = user.validation_item if split == "validation" else user.test_item
        seen = set(history)
        unreachable += target not in item_set
        target_in_history += target in seen

        started = perf_counter()
        popularity_hits = popularity.search(top_k=max_k, exclude_seen_items=seen)
        accumulators["popularity"]["latency_ms"].append((perf_counter() - started) * 1000)
        started = perf_counter()
        user_vector = mean_user_embedding(history, retriever)
        if user_vector is None:
            dense_hits: list[SearchHit] = []
            accumulators["dense"]["no_history_embedding"] += 1
        else:
            dense_hits = retriever.search(user_vector, top_k=max_k, exclude_seen_items=seen)
        accumulators["dense"]["latency_ms"].append((perf_counter() - started) * 1000)

        for method, hits in (("popularity", popularity_hits), ("dense", dense_hits)):
            predicted = [hit.item_id for hit in hits]
            for k in k_values:
                accumulators[method]["metrics"][f"Recall@{k}"].append(
                    recall_at_k(predicted, {target}, k)
                )
                accumulators[method]["ndcg"][f"NDCG@{k}"].append(
                    ndcg_at_k(predicted, {target}, k)
                )

    result: dict[str, Any] = {
        "split": split,
        "users": len(histories),
        "items": len(retriever.item_ids),
        "unreachable_targets": unreachable,
        "target_already_in_history": target_in_history,
        "history_policy": "train" if split == "validation" else "train_plus_validation",
        "methods": {},
    }
    for method, values in accumulators.items():
        result["methods"][method] = {
            **{key: mean(scores) if scores else 0.0 for key, scores in values["metrics"].items()},
            **{key: mean(scores) if scores else 0.0 for key, scores in values["ndcg"].items()},
            "avg_retrieval_latency_ms": mean(values["latency_ms"])
            if values["latency_ms"] else 0.0,
            "no_history_embedding": values["no_history_embedding"],
        }
    return result
