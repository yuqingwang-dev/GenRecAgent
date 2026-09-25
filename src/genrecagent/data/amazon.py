"""Safe normalization and chronological leave-last-out for Amazon reviews."""

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from genrecagent.data.models import Interaction, Item, UserHistory


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _text_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(text for entry in value if (text := _text(entry)))


def _number(value: Any, *, minimum: float = 0.0) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= minimum else None


def normalize_item(row: Mapping[str, Any]) -> Item | None:
    item_id = _text(row.get("parent_asin"))
    if not item_id:
        return None
    description = " ".join(_text_list(row.get("description")))
    categories = _text_list(row.get("categories"))
    main_category = _text(row.get("main_category"))
    if main_category and main_category not in categories:
        categories = (main_category, *categories)
    return Item(
        item_id=item_id,
        title=_text(row.get("title")),
        description=description,
        features=_text_list(row.get("features")),
        categories=categories,
        price=_number(row.get("price")),
        average_rating=_number(row.get("average_rating"), minimum=0.0),
    )


def normalize_interaction(row: Mapping[str, Any], source_index: int) -> Interaction | None:
    user_id = _text(row.get("user_id"))
    item_id = _text(row.get("parent_asin"))
    timestamp = row.get("timestamp")
    if not user_id or not item_id or isinstance(timestamp, bool):
        return None
    try:
        timestamp = int(timestamp)
    except (TypeError, ValueError, OverflowError):
        return None
    if timestamp <= 0:
        return None
    verified = row.get("verified_purchase")
    return Interaction(
        user_id=user_id,
        item_id=item_id,
        rating=_number(row.get("rating")),
        timestamp=timestamp,
        verified_purchase=verified if isinstance(verified, bool) else None,
        source_index=source_index,
    )


def build_item_text(item: Item) -> str:
    """Produce metadata-only text; never incorporate user reviews or targets."""

    fields = (
        ("Title", item.title),
        ("Category", " > ".join(item.categories)),
        ("Features", "; ".join(item.features)),
        ("Description", item.description),
    )
    return "\n".join(f"{label}: {value}" for label, value in fields if value.strip())


@dataclass(slots=True)
class PreparedData:
    items: dict[str, Item]
    interactions: list[Interaction]
    histories: list[UserHistory]
    stats: dict[str, int | str | None]


def _sample_key(seed: int, user_id: str) -> str:
    return hashlib.sha256(f"{seed}:{user_id}".encode()).hexdigest()


def prepare_records(
    metadata_rows: Iterable[Mapping[str, Any]],
    review_rows: Iterable[Mapping[str, Any]],
    *,
    min_user_interactions: int = 3,
    max_users: int | None = None,
    seed: int = 42,
    progress_every: int | None = None,
) -> PreparedData:
    if min_user_interactions < 3:
        raise ValueError("leave-last-out requires at least three interactions")
    if max_users is not None and max_users < 1:
        raise ValueError("max_users must be positive")

    stats: Counter[str] = Counter()
    items: dict[str, Item] = {}
    for row in metadata_rows:
        stats["raw_metadata_rows"] += 1
        item = normalize_item(row)
        if item is None:
            stats["metadata_missing_parent_asin"] += 1
        elif item.item_id in items:
            stats["duplicate_metadata_rows"] += 1
        else:
            items[item.item_id] = item
        if progress_every and stats["raw_metadata_rows"] % progress_every == 0:
            print(f"metadata: {stats['raw_metadata_rows']} rows", flush=True)

    first_interactions: dict[tuple[str, str], Interaction] = {}
    for index, row in enumerate(review_rows):
        stats["raw_review_rows"] += 1
        if progress_every and stats["raw_review_rows"] % progress_every == 0:
            print(f"reviews: {stats['raw_review_rows']} rows", flush=True)
        interaction = normalize_interaction(row, index)
        if interaction is None:
            stats["invalid_review_rows"] += 1
            continue
        if interaction.item_id not in items:
            stats["reviews_without_metadata"] += 1
            continue
        key = (interaction.user_id, interaction.item_id)
        previous = first_interactions.get(key)
        if previous is not None:
            stats["duplicate_user_item_reviews"] += 1
        if previous is None or (interaction.timestamp, index) < (
            previous.timestamp,
            previous.source_index,
        ):
            first_interactions[key] = interaction

    user_counts = Counter(interaction.user_id for interaction in first_interactions.values())
    candidate_users = {
        user_id
        for user_id, count in user_counts.items()
        if count >= min_user_interactions
    }
    stats["users_below_min_interactions"] = len(user_counts) - len(candidate_users)
    by_user: dict[str, list[Interaction]] = defaultdict(list)
    for interaction in first_interactions.values():
        if interaction.user_id in candidate_users:
            by_user[interaction.user_id].append(interaction)
    del first_interactions, user_counts, candidate_users
    eligible: dict[str, list[Interaction]] = {}
    for user_id, sequence in by_user.items():
        sequence.sort(key=lambda row: (row.timestamp, row.source_index))
        stats["equal_timestamp_adjacent_pairs"] += sum(
            first.timestamp == second.timestamp
            for first, second in zip(sequence, sequence[1:], strict=False)
        )
        eligible[user_id] = sequence

    stats["eligible_users_before_sampling"] = len(eligible)
    selected_ids = sorted(eligible, key=lambda user_id: (_sample_key(seed, user_id), user_id))
    if max_users is not None:
        selected_ids = selected_ids[:max_users]
    histories: list[UserHistory] = []
    interactions: list[Interaction] = []
    for user_id in sorted(selected_ids):
        sequence = eligible[user_id]
        train, validation, test = sequence[:-2], sequence[-2], sequence[-1]
        histories.append(
            UserHistory(
                user_id=user_id,
                train_items=tuple(row.item_id for row in train),
                validation_item=validation.item_id,
                test_item=test.item_id,
                train_timestamps=tuple(row.timestamp for row in train),
                validation_timestamp=validation.timestamp,
                test_timestamp=test.timestamp,
            )
        )
        interactions.extend(sequence)

    summary: dict[str, int | str | None] = dict(stats)
    summary.update(
        {
            "seed": seed,
            "max_users": max_users,
            "sampling": "SHA256(seed:user_id), smallest hashes first",
            "items_in_catalog": len(items),
            "items_missing_price": sum(item.price is None for item in items.values()),
            "users": len(histories),
            "interactions": len(interactions),
            "train_interactions": sum(len(user.train_items) for user in histories),
            "validation_interactions": len(histories),
            "test_interactions": len(histories),
        }
    )
    return PreparedData(items=items, interactions=interactions, histories=histories, stats=summary)
