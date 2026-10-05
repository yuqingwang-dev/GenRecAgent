"""Ground model price parameters in explicit request text before execution."""

import re
from dataclasses import dataclass

from genrecagent.agent.constraints import PriceLimit
from genrecagent.agent.intent import ParsedIntent, parse_recommendation_intent

_AMOUNT = r"(?P<amount>-?\d+(?:\.\d+)?)(?!(?:\d|\.\d))"
_PATTERNS = (
    re.compile(
        r"(?P<op>at most|no more than|not more than|up to|under|below|less than)"
        r"\s*\$?" + _AMOUNT + r"(?:\s*(?:usd|dollars?))?", re.IGNORECASE,
    ),
    re.compile(
        r"(?:价格|单价|每件|预算)?\s*(?P<op>不超过|至多|最多|不高于|低于|少于)"
        r"\s*\$?" + _AMOUNT + r"\s*(?:美元)?",
    ),
    re.compile(_AMOUNT + r"\s*美元\s*(?P<op>以内)"),
)
_INCLUSIVE = {
    "at most", "no more than", "not more than", "up to", "不超过", "至多", "最多", "不高于", "以内"
}
_BUDGET_MARKERS = re.compile(
    r"\b(?:price|budget|under|below|less than|at most|no more than|not more than|up to)\b"
    r"|\$|价格|预算|不超过|至多|最多|不高于|低于|少于|以内", re.IGNORECASE,
)
_UNSUPPORTED_CURRENCY = re.compile(
    r"人民币|\d\s*元|￥|¥|€|£|\b(?:rmb|cny|eur|gbp)\b", re.IGNORECASE,
)
_UNSUPPORTED_CONDITIONS = re.compile(
    r"\b(?:vegan|cruelty[- ]free|in[- ]stock|fragrance[- ]free|hypoallergenic|"
    r"shampoo|moisturizer|sunscreen|shipping|delivery|total)\b"
    r"|库存|现货|纯素|无香|洗发|防晒|保湿|运费|到货|总价|合计", re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class BudgetObservation:
    status: str
    limit: PriceLimit | None = None
    evidence: str | None = None


def observe_budget(text: str) -> BudgetObservation:
    if _UNSUPPORTED_CURRENCY.search(text):
        return BudgetObservation("unsupported_currency")
    matches = [match for pattern in _PATTERNS for match in pattern.finditer(text)]
    if len(matches) > 1:
        return BudgetObservation("invalid_budget")
    if not matches:
        return BudgetObservation("invalid_budget" if _BUDGET_MARKERS.search(text) else "none")
    match = matches[0]
    try:
        limit = PriceLimit(float(match["amount"]), match["op"].lower() in _INCLUSIVE)
    except ValueError:
        return BudgetObservation("invalid_budget")
    return BudgetObservation("verified", limit, match[0])


def guard_policy_decision(text: str, parsed: ParsedIntent, evidence: str | None) -> None:
    observation = observe_budget(text)
    if observation.status == "unsupported_currency":
        if parsed.status != "unsupported_request":
            raise ValueError("unsupported_currency_was_not_rejected")
    elif observation.status == "invalid_budget":
        if parsed.status != "invalid_request":
            raise ValueError("invalid_or_ungrounded_budget_was_not_rejected")
    elif parsed.status == "accepted":
        if parsed.price_limit != observation.limit:
            raise ValueError("budget_omitted_changed_or_invented")
        if observation.limit is not None and (
            not evidence or evidence not in text
            or observe_budget(evidence).status != "verified"
            or observe_budget(evidence).limit != observation.limit
        ):
            raise ValueError("budget_evidence_is_not_a_matching_request_span")
    if parsed.status == "accepted":
        if _UNSUPPORTED_CONDITIONS.search(text):
            raise ValueError("unsupported_condition_was_accepted")
        if parsed.intent == "generative" and not re.search(
            r"generative|生成式", text, re.IGNORECASE
        ):
            raise ValueError("generative_route_was_not_explicitly_requested")
    rules = parse_recommendation_intent(text)
    if rules.status == "accepted" and (
        parsed.status != rules.status or parsed.intent != rules.intent
        or parsed.price_limit != rules.price_limit
    ):
        raise ValueError("decision_conflicts_with_recognized_request")
