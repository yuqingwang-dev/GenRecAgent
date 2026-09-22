"""Small persistence-ready semantic codebook placeholder."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class SemanticCodebook:
    """Map item identifiers to semantic token sequences in a future phase."""

    assignments: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def encode(self, item_id: str) -> tuple[str, ...]:
        if item_id not in self.assignments:
            raise KeyError(f"No semantic ID assigned for {item_id}")
        return self.assignments[item_id]

    def add(self, item_id: str, tokens: tuple[str, ...]) -> None:
        self.assignments[item_id] = tokens

