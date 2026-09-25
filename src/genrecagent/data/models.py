"""Normalized records; raw dataset dictionaries stop at the data boundary."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Item:
    item_id: str
    title: str
    description: str
    features: tuple[str, ...]
    categories: tuple[str, ...]
    price: float | None
    average_rating: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Interaction:
    user_id: str
    item_id: str
    rating: float | None
    timestamp: int
    verified_purchase: bool | None
    source_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UserHistory:
    user_id: str
    train_items: tuple[str, ...]
    validation_item: str
    test_item: str
    train_timestamps: tuple[int, ...]
    validation_timestamp: int
    test_timestamp: int

    @property
    def validation_history(self) -> tuple[str, ...]:
        return self.train_items

    @property
    def test_history(self) -> tuple[str, ...]:
        return (*self.train_items, self.validation_item)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
