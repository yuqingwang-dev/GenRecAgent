"""Aggregate validation-only code and item retrieval failure modes."""

from dataclasses import dataclass
from math import log2

from genrecagent.retrieval.generative import GenerativeSearchResult
from genrecagent.retrieval.semantic_trie import SemanticID


@dataclass(slots=True)
class RetrievalBucket:
    users: int = 0
    hits_5: int = 0
    hits_10: int = 0
    hits_20: int = 0
    ndcg_20_sum: float = 0.0
    target_code_found: int = 0
    code_found_item_missed: int = 0

    def add(
        self, target_item: str, target_code: SemanticID,
        outcome: GenerativeSearchResult,
    ) -> None:
        self.users += 1
        rank = next(
            (hit.rank for hit in outcome.hits if hit.item_id == target_item), None
        )
        code_found = any(
            prediction.semantic_id == target_code for prediction in outcome.predicted_codes
        )
        self.hits_5 += rank is not None and rank <= 5
        self.hits_10 += rank is not None and rank <= 10
        self.hits_20 += rank is not None and rank <= 20
        if rank is not None and rank <= 20:
            self.ndcg_20_sum += 1.0 / log2(rank + 1)
        self.target_code_found += code_found
        self.code_found_item_missed += code_found and (rank is None or rank > 20)

    def summary(self) -> dict[str, int | float]:
        if not self.users:
            raise ValueError("Cannot summarize an empty cohort")
        return {
            "users": self.users,
            "Recall@5": self.hits_5 / self.users,
            "Recall@10": self.hits_10 / self.users,
            "Recall@20": self.hits_20 / self.users,
            "NDCG@20": self.ndcg_20_sum / self.users,
            "CodeRecall@20": self.target_code_found / self.users,
            "code_not_predicted": self.users - self.target_code_found,
            "code_predicted_item_missed": self.code_found_item_missed,
            "item_hit": self.hits_20,
        }


def validation_cohorts(
    *, history_length: int, target_code_seen: bool, collision_size: int,
) -> tuple[tuple[str, str], ...]:
    if history_length < 1 or collision_size < 1:
        raise ValueError("History and target collision group must be nonempty")
    return (
        ("all", "all"),
        ("history_length", "1" if history_length == 1 else (
            "2" if history_length == 2 else "3_plus"
        )),
        ("target_code_seen_in_train", "yes" if target_code_seen else "no"),
        ("target_collision_size", "1" if collision_size == 1 else (
            "2_to_4" if collision_size <= 4 else "5_plus"
        )),
    )
