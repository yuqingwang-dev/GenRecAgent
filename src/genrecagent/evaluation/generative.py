"""Shared item-level evaluation for code-based recommendation routes."""

from collections import Counter
from dataclasses import asdict
from statistics import mean
from time import perf_counter

from genrecagent.evaluation.recommendation import ndcg_at_k, recall_at_k
from genrecagent.retrieval.generative import GenerativeRetriever


def evaluate_generative(
    examples: list[dict],
    retriever: GenerativeRetriever,
    k_values: list[int],
    train_target_codes: set[tuple[str, ...]],
    split: str,
) -> dict[str, object]:
    if not k_values or min(k_values) < 1:
        raise ValueError("Evaluation K values must be positive")
    results: dict[str, list[float]] = {
        **{f"Recall@{k}": [] for k in k_values},
        **{f"NDCG@{k}": [] for k in k_values},
    }
    diagnostics: Counter[str] = Counter()
    latencies: list[float] = []
    seen_target_codes = 0
    for number, row in enumerate(examples, start=1):
        target = row["target_item"]
        history = row["history_items"]
        seen_target_codes += tuple(row["target_code"]) in train_target_codes
        started = perf_counter()
        outcome = retriever.retrieve(history, top_k=max(k_values))
        latencies.append((perf_counter() - started) * 1000)
        predicted = [hit.item_id for hit in outcome.hits]
        for k in k_values:
            results[f"Recall@{k}"].append(recall_at_k(predicted, {target}, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(predicted, {target}, k))
        for key, value in asdict(outcome.diagnostics).items():
            diagnostics[key] += value
        if number % 500 == 0:
            print(f"{split}: evaluated {number}/{len(examples)} users", flush=True)
    count = len(examples)
    return {
        "users": count,
        "target_code_seen_in_train_fraction": seen_target_codes / count if count else 0.0,
        **{key: mean(values) if values else 0.0 for key, values in results.items()},
        "avg_retrieval_latency_ms": mean(latencies) if latencies else 0.0,
        "diagnostic_totals": dict(diagnostics),
        "diagnostic_averages": {
            key: value / count if count else 0.0 for key, value in diagnostics.items()
        },
    }
