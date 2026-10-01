"""Item-to-learned-token mapping that preserves quantization collisions."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SemanticCodebook:
    """Map item IDs to codes; reverse lookup returns every colliding item."""

    assignments: dict[str, tuple[str, ...]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    _reverse: dict[tuple[str, ...], set[str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for item_id, tokens in self.assignments.items():
            self._reverse.setdefault(tokens, set()).add(item_id)

    def encode(self, item_id: str) -> tuple[str, ...]:
        if item_id not in self.assignments:
            raise KeyError(f"No semantic ID assigned for {item_id}")
        return self.assignments[item_id]

    def add(self, item_id: str, tokens: tuple[str, ...]) -> None:
        if not item_id or not tokens:
            raise ValueError("Item ID and learned token sequence must be nonempty")
        old_tokens = self.assignments.get(item_id)
        if old_tokens is not None:
            old_group = self._reverse[old_tokens]
            old_group.remove(item_id)
            if not old_group:
                del self._reverse[old_tokens]
        self.assignments[item_id] = tokens
        self._reverse.setdefault(tokens, set()).add(item_id)

    def lookup(self, tokens: tuple[str, ...]) -> tuple[str, ...]:
        """Do not silently select one item when Semantic IDs collide."""

        return tuple(sorted(self._reverse.get(tokens, ())))

    def collision_stats(self) -> dict[str, int]:
        colliding = [len(group) for group in self._reverse.values() if len(group) > 1]
        return {
            "item_count": len(self.assignments),
            "unique_code_count": len(self._reverse),
            "collision_group_count": len(colliding),
            "colliding_item_count": sum(colliding),
            "largest_collision_group": max(colliding, default=0),
        }

    def save(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "assignments": {
                item_id: list(self.assignments[item_id]) for item_id in sorted(self.assignments)
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SemanticCodebook":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            assignments={
                item_id: tuple(tokens) for item_id, tokens in payload["assignments"].items()
            },
            metadata=payload.get("metadata", {}),
        )
