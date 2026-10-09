"""Tokenizer quality gates independent of recommendation targets or test hits."""

import math

import numpy as np


def code_statistics(codes: np.ndarray, codebook_size: int) -> dict:
    values = np.asarray(codes)
    if (
        values.ndim != 2
        or not len(values)
        or not np.issubdtype(values.dtype, np.integer)
        or values.min() < 0
        or values.max() >= codebook_size
    ):
        raise ValueError("Expected nonempty integer code matrix inside the declared codebook")
    usage = []
    for level in range(values.shape[1]):
        counts = np.bincount(values[:, level], minlength=codebook_size)
        probabilities = counts[counts > 0] / len(values)
        usage.append(
            {
                "used_codes": int(np.count_nonzero(counts)),
                "codebook_size": codebook_size,
                "utilization": float(np.count_nonzero(counts) / codebook_size),
                "perplexity": float(math.exp(-np.sum(probabilities * np.log(probabilities)))),
            }
        )
    _, groups = np.unique(values, axis=0, return_counts=True)
    return {
        "codebook_usage_by_level": usage,
        "unique_code_count": len(groups),
        "unique_ratio": len(groups) / len(values),
        "largest_group_fraction": int(groups.max()) / len(values),
    }


def quality_gate(
    stats: dict,
    *,
    min_utilization: float = 0,
    min_perplexity_fraction: float = 0,
    min_unique_ratio: float = 0,
    max_group_fraction: float = 1,
    max_reconstruction_mse: float | None = None,
) -> dict:
    enabled = bool(
        min_utilization
        or min_perplexity_fraction
        or min_unique_ratio
        or max_group_fraction < 1
        or max_reconstruction_mse is not None
    )
    checks = {
        "per_level_utilization": all(
            row["utilization"] >= min_utilization for row in stats["codebook_usage_by_level"]
        ),
        "per_level_perplexity": all(
            row["perplexity"] >= min_perplexity_fraction * row["codebook_size"]
            for row in stats["codebook_usage_by_level"]
        ),
        "unique_ratio": stats["unique_ratio"] >= min_unique_ratio,
        "largest_group_fraction": stats["largest_group_fraction"] <= max_group_fraction,
        "reconstruction_mse": max_reconstruction_mse is None
        or (stats["reconstruction_mse"] <= max_reconstruction_mse),
    }
    return {
        "enabled": enabled,
        "passed": all(checks.values()) if enabled else None,
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
    }


def checkpoint_score(stats: dict, gate: dict, mse: float) -> tuple:
    if not math.isfinite(mse):
        raise ValueError("Nonfinite validation reconstruction cannot select a checkpoint")
    if not gate["enabled"] or gate["passed"]:
        return (1, -mse, 0, 0)
    return (
        0,
        min(row["utilization"] for row in stats["codebook_usage_by_level"]),
        stats["unique_ratio"],
        -mse,
    )
