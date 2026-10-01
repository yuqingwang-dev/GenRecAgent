"""Leakage-aware next-item examples from chronological user histories."""

from dataclasses import asdict, dataclass
from typing import Literal

from genrecagent.data.models import UserHistory
from genrecagent.semantic_id.codebook import SemanticCodebook

Split = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class SequenceExample:
    user_id: str
    split: Split
    history_items: tuple[str, ...]
    target_item: str
    history_codes: tuple[tuple[str, ...], ...]
    target_code: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_sequence_examples(
    histories: list[UserHistory],
    codebook: SemanticCodebook,
    *,
    max_history: int = 20,
) -> dict[Split, list[SequenceExample]]:
    """Train only on train prefixes; reserve final two events for evaluation."""

    if max_history < 1:
        raise ValueError("max_history must be positive")
    result: dict[Split, list[SequenceExample]] = {
        "train": [], "validation": [], "test": [],
    }
    seen_users: set[str] = set()

    def append(split: Split, user_id: str, history: tuple[str, ...], target: str) -> None:
        context = history[-max_history:]
        result[split].append(
            SequenceExample(
                user_id=user_id,
                split=split,
                history_items=context,
                target_item=target,
                history_codes=tuple(codebook.encode(item_id) for item_id in context),
                target_code=codebook.encode(target),
            )
        )

    for user in sorted(histories, key=lambda row: row.user_id):
        if user.user_id in seen_users:
            raise ValueError(f"Duplicate user history: {user.user_id}")
        seen_users.add(user.user_id)
        if len(user.train_items) != len(user.train_timestamps) or not user.train_items:
            raise ValueError(f"Invalid training history: {user.user_id}")
        if tuple(sorted(user.train_timestamps)) != user.train_timestamps:
            raise ValueError(f"Training timestamps are not ordered: {user.user_id}")
        if user.train_timestamps[-1] > user.validation_timestamp:
            raise ValueError(f"Validation precedes training: {user.user_id}")
        if user.validation_timestamp > user.test_timestamp:
            raise ValueError(f"Test precedes validation: {user.user_id}")
        for target_index in range(1, len(user.train_items)):
            append(
                "train", user.user_id,
                user.train_items[:target_index], user.train_items[target_index],
            )
        append("validation", user.user_id, user.validation_history, user.validation_item)
        append("test", user.user_id, user.test_history, user.test_item)
    return result
