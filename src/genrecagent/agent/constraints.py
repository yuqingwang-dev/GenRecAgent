"""Static metadata price limits; missing values never prove budget compliance."""

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real


def known_price(value: object) -> bool:
    return (
        isinstance(value, Real) and not isinstance(value, bool)
        and isfinite(float(value)) and value >= 0
    )


@dataclass(frozen=True, slots=True)
class PriceLimit:
    amount: float
    inclusive: bool = True

    def __post_init__(self) -> None:
        if not known_price(self.amount) or not isinstance(self.inclusive, bool):
            raise ValueError("Price limit requires a finite nonnegative amount and boolean flag")

    def allows(self, price: object) -> bool:
        if not known_price(price):
            return False
        return price <= self.amount if self.inclusive else price < self.amount

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)
