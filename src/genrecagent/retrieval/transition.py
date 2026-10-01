"""Train-only count baseline for next-item Semantic ID token prediction."""

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from genrecagent.retrieval.semantic_trie import HistoryCodes, SemanticID
from genrecagent.semantic_id.codebook import SemanticCodebook


class CodeTransitionScorer:
    """Smoothed last-code transition plus target-prefix counts.

    This baseline is not a Transformer. All counts come from train examples only.
    """

    def __init__(self, codebook: SemanticCodebook, *, alpha: float = 10.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        codes = set(codebook.assignments.values())
        lengths = {len(code) for code in codes}
        if len(lengths) != 1 or not codes:
            raise ValueError("Catalog codes must have one fixed positive length")
        self.depth = lengths.pop()
        self.valid_codes = frozenset(codes)
        self.vocabulary = tuple(
            tuple(sorted({code[level] for code in codes})) for level in range(self.depth)
        )
        self.alpha = alpha
        self.global_counts = [Counter() for _ in range(self.depth)]
        self.first_transition: dict[str, Counter[str]] = defaultdict(Counter)
        self.prefix_counts: dict[SemanticID, Counter[str]] = defaultdict(Counter)
        self.train_examples = 0
        self._cache: dict[tuple[str, SemanticID], dict[str, float]] = {}

    def fit(self, examples: Iterable[Mapping[str, Any]]) -> "CodeTransitionScorer":
        for example in examples:
            if example.get("split") != "train":
                raise ValueError("Transition scorer may fit only train examples")
            target = tuple(example["target_code"])
            history = example["history_codes"]
            if target not in self.valid_codes or any(
                token not in self.vocabulary[level] for level, token in enumerate(target)
            ):
                raise ValueError("Target code does not belong to the catalog vocabulary")
            if not history or len(history[-1]) != self.depth:
                raise ValueError("A nonempty coded history is required")
            self.first_transition[history[-1][0]][target[0]] += 1
            for level, token in enumerate(target):
                self.global_counts[level][token] += 1
                if level:
                    self.prefix_counts[target[:level]][token] += 1
            self.train_examples += 1
        if self.train_examples == 0:
            raise ValueError("No train examples were provided")
        self._cache.clear()
        return self

    def log_probabilities(
        self, history: HistoryCodes, prefix: SemanticID
    ) -> Mapping[str, float]:
        if not self.train_examples:
            raise RuntimeError("Fit the transition scorer before decoding")
        level = len(prefix)
        if level >= self.depth:
            return {}
        last = history[-1][0] if history else "<no-history>"
        key = (last if level == 0 else "", prefix)
        if key in self._cache:
            return self._cache[key]
        vocabulary = self.vocabulary[level]
        global_counts = self.global_counts[level]
        global_total = sum(global_counts.values())
        prior_denominator = global_total + len(vocabulary)
        conditional = self.first_transition.get(last, Counter()) if level == 0 else (
            self.prefix_counts.get(prefix, Counter())
        )
        conditional_total = sum(conditional.values())
        denominator = conditional_total + self.alpha
        scores = {
            token: math.log(
                (
                    conditional[token]
                    + self.alpha * (global_counts[token] + 1) / prior_denominator
                ) / denominator
            )
            for token in vocabulary
        }
        self._cache[key] = scores
        return scores

    def save(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(path)
        if not self.train_examples:
            raise RuntimeError("Fit the scorer before saving")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "train_only_code_transition_baseline",
            "alpha": self.alpha,
            "vocabulary": self.vocabulary,
            "train_examples": self.train_examples,
            "global_counts": [dict(counts) for counts in self.global_counts],
            "first_transition": {
                source: dict(counts) for source, counts in self.first_transition.items()
            },
            "prefix_counts": [
                {"prefix": list(prefix), "counts": dict(counts)}
                for prefix, counts in sorted(self.prefix_counts.items())
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, codebook: SemanticCodebook) -> "CodeTransitionScorer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["model_type"] != "train_only_code_transition_baseline":
            raise ValueError("Unexpected transition model type")
        model = cls(codebook, alpha=float(payload["alpha"]))
        if tuple(tuple(tokens) for tokens in payload["vocabulary"]) != model.vocabulary:
            raise ValueError("Saved vocabulary does not match the catalog")
        model.train_examples = int(payload["train_examples"])
        model.global_counts = [Counter(row) for row in payload["global_counts"]]
        model.first_transition = defaultdict(
            Counter, {source: Counter(row) for source, row in payload["first_transition"].items()}
        )
        model.prefix_counts = defaultdict(
            Counter,
            {
                tuple(row["prefix"]): Counter(row["counts"])
                for row in payload["prefix_counts"]
            },
        )
        return model
