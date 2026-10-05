"""Conservative English/Chinese grammar for three explicit recommendation intents."""

import re
from dataclasses import dataclass
from typing import Literal

from genrecagent.agent.constraints import PriceLimit

IntentName = Literal["personalized", "popular", "generative"]
ParseStatus = Literal["accepted", "invalid_request", "unsupported_request"]
_CURRENCY = re.compile(r"人民币|\d\s*元|￥|\b(?:rmb|cny|eur)\b|€", re.IGNORECASE)
_MARKERS = re.compile(
    r"\b(?:price|budget|under|below|less than|at most|no more than)\b"
    r"|价格|预算|不超过|至多|低于|少于|以内", re.IGNORECASE,
)
_BUDGETS = (
    re.compile(
        r"(?P<op>at most|no more than|under|below|less than)\s*\$?"
        r"(?P<amount>\d+(?:\.\d{1,2})?)(?:\s*(?:usd|dollars?))?$", re.IGNORECASE,
    ),
    re.compile(
        r"(?:价格)?\s*(?P<op>不超过|至多|低于|少于)\s*\$?"
        r"(?P<amount>\d+(?:\.\d{1,2})?)\s*(?:美元)?$",
    ),
)
_INTENTS = {
    "personalized": (
        re.compile(
            r"(?:recommend(?: products)?|(?:give me )?recommendations)\s+"
            r"(?:based on|from|using)\s+my history", re.IGNORECASE,
        ),
        re.compile(r"(?:根据|基于)(?:我的)?(?:浏览|交互)?历史(?:给我)?推荐(?:商品)?"),
    ),
    "popular": (
        re.compile(r"(?:show|recommend|give me)\s+(?:popular products|bestsellers)", re.IGNORECASE),
        re.compile(r"(?:推荐|给我|看看)(?:热门|畅销)(?:商品)?"),
    ),
    "generative": (
        re.compile(r"(?:use|try)\s+generative\s+(?:retrieval|recommendations?)", re.IGNORECASE),
        re.compile(r"(?:使用|尝试|用)生成式(?:检索)?(?:推荐(?:商品)?)?"),
    ),
}


@dataclass(frozen=True, slots=True)
class ParsedIntent:
    status: ParseStatus
    intent: IntentName | None = None
    price_limit: PriceLimit | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status, "intent": self.intent,
            "price_limit": self.price_limit.to_dict() if self.price_limit else None,
            "reason": self.reason,
        }


def parse_recommendation_intent(text: str) -> ParsedIntent:
    if not isinstance(text, str) or not text.strip():
        return ParsedIntent("invalid_request", reason="empty_request")
    body = text.strip().rstrip(".。!！?？").strip()
    if _CURRENCY.search(body):
        return ParsedIntent("unsupported_request", reason="unsupported_currency")
    price_limit = None
    for pattern in _BUDGETS:
        match = pattern.search(body)
        if match:
            try:
                price_limit = PriceLimit(
                    float(match["amount"]),
                    inclusive=match["op"].lower() in {"at most", "no more than", "不超过", "至多"},
                )
            except ValueError:
                return ParsedIntent("invalid_request", reason="invalid_price_limit")
            body = body[:match.start()].rstrip(" ,;，；")
            if _MARKERS.search(body):
                return ParsedIntent("invalid_request", reason="multiple_or_conflicting_budgets")
            break
    else:
        if _MARKERS.search(body):
            return ParsedIntent("invalid_request", reason="unrecognized_price_limit")
    for intent, patterns in _INTENTS.items():
        if any(pattern.fullmatch(body) for pattern in patterns):
            return ParsedIntent("accepted", intent=intent, price_limit=price_limit)
    return ParsedIntent("unsupported_request", reason="unsupported_intent_or_condition")
