"""Schema-checked local LLM intent decisions with bounded repair and rule fallback."""

import hashlib
import json
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from genrecagent.agent.constraints import PriceLimit
from genrecagent.agent.intent import ParsedIntent, parse_recommendation_intent
from genrecagent.agent.policy_guard import guard_policy_decision, observe_budget
from genrecagent.llm.base import BaseLLM

TOOL_DEFINITION = {
    "name": "recommendation_policy",
    "description": "Compile the entire request; reject conditions the catalog cannot support.",
    "parameters": {
        "type": "object", "additionalProperties": False,
        "required": ["status", "intent", "price_limit", "reason"],
        "properties": {
            "status": {"type": "string",
                       "enum": ["accepted", "invalid_request", "unsupported_request"]},
            "intent": {"type": ["string", "null"],
                       "enum": ["personalized", "popular", "generative", None]},
            "price_limit": {
                "anyOf": [
                    {"type": "null"},
                    {"type": "object", "additionalProperties": False,
                     "required": ["amount", "inclusive", "evidence"],
                     "properties": {
                         "amount": {"type": "number", "minimum": 0},
                         "inclusive": {"type": "boolean"},
                         "evidence": {"type": "string", "minLength": 1},
                     }},
                ],
            },
            "reason": {"type": ["string", "null"]},
        },
    },
}
POLICY_PROMPT = """Supported capabilities:
- personalized: recommendations based on the user's observed history.
- popular: explicit popular, trending or bestseller requests.
- generative: only when the user explicitly asks for generative retrieval.
- Optional per-item maximum price in USD/source metadata price units.
Accept only requests that these capabilities fully cover. Product-specific searches (shampoo,
moisturizer, etc.), ingredients/vegan/cruelty-free, stock, shipping, category, total basket budgets,
and non-USD currencies are unsupported_request. Negative, NaN/Infinity, malformed or multiple
conflicting budgets are invalid_request. Do not discard additional conditions.
Requests combining popular and generative routes are unsupported_request: one intent is allowed.
For accepted requests, intent is one supported name; reason may be null. No budget means
price_limit=null. A budget requires amount, inclusive, and evidence copied verbatim from the
request. Evidence must be a continuous phrase containing BOTH the comparison and the numeric
amount, for example "below $12", not just "below" and never a word absent from the request.
under/below/less than/低于/少于 are exclusive.
at most/no more than/up to/不超过/至多/最多/以内 are inclusive.
For rejected requests, intent and price_limit must both be null and reason explains
the rejection in a brief phrase. Return one recommendation_policy function call with name and
arguments only.
Examples:
Input: Show popular products below $12.
Output: {"name":"recommendation_policy","arguments":{
"status":"accepted","intent":"popular",
"price_limit":{"amount":12,"inclusive":false,"evidence":"below $12"},"reason":null}}
Input: 根据我的历史推荐商品，单价最多8美元。
Output: {"name":"recommendation_policy","arguments":{
"status":"accepted","intent":"personalized",
"price_limit":{"amount":8,"inclusive":true,"evidence":"单价最多8美元"},"reason":null}}
Input: Suggest popular products.
Output: {"name":"recommendation_policy","arguments":{
"status":"accepted","intent":"popular","price_limit":null,"reason":null}}
Input: Use my history and keep prices under -2 dollars.
Output: {"name":"recommendation_policy","arguments":{
"status":"invalid_request","intent":null,"price_limit":null,"reason":"negative_budget"}}
Input: Show only vegan products.
Output: {"name":"recommendation_policy","arguments":{
"status":"unsupported_request","intent":null,"price_limit":null,
"reason":"ingredient_filter_unsupported"}}
"""
PROMPT_SHA256 = hashlib.sha256(POLICY_PROMPT.encode()).hexdigest()
SCHEMA_SHA256 = hashlib.sha256(json.dumps(TOOL_DEFINITION, sort_keys=True).encode()).hexdigest()


def validate_policy_call(call: dict[str, Any]) -> tuple[ParsedIntent, str | None]:
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        raise ValueError("tool_envelope_keys_invalid")
    if call["name"] != TOOL_DEFINITION["name"]:
        raise ValueError("unknown_tool_name")
    args = call["arguments"]
    if not isinstance(args, dict) or set(args) != {"status", "intent", "price_limit", "reason"}:
        raise ValueError("tool_argument_keys_invalid")
    status, intent, budget, reason = (
        args["status"], args["intent"], args["price_limit"], args["reason"]
    )
    if not isinstance(status, str) or status not in {
        "accepted", "invalid_request", "unsupported_request"
    }:
        raise ValueError("invalid_status")
    if reason is not None and (not isinstance(reason, str) or len(reason) > 256):
        raise ValueError("invalid_reason")
    if status != "accepted":
        if intent is not None or budget is not None or not reason:
            raise ValueError("rejection_arguments_must_have_null_intent_and_budget")
        return ParsedIntent(status, reason=reason), None
    if not isinstance(intent, str) or intent not in {"personalized", "popular", "generative"}:
        raise ValueError("invalid_intent")
    limit, evidence = None, None
    if budget is not None:
        if not isinstance(budget, dict) or set(budget) != {"amount", "inclusive", "evidence"}:
            raise ValueError(
                "price_argument_keys_invalid: require amount, inclusive, evidence only"
            )
        limit = PriceLimit(budget["amount"], budget["inclusive"])
        evidence = budget["evidence"]
        if not isinstance(evidence, str) or not 1 <= len(evidence) <= 256:
            raise ValueError("invalid_price_evidence")
    return ParsedIntent(status, intent=intent, price_limit=limit, reason=reason), evidence


@dataclass(frozen=True, slots=True)
class PolicyAttempt:
    number: int
    call: dict[str, Any] | None
    parsed: ParsedIntent | None
    schema_valid: bool
    guard_valid: bool
    error: str | None
    generation: dict[str, object] | None
    latency_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "number": self.number, "call": self.call,
            "parsed": self.parsed.to_dict() if self.parsed else None,
            "schema_valid": self.schema_valid, "guard_valid": self.guard_valid,
            "error": self.error, "generation": self.generation, "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    parsed: ParsedIntent
    attempts: tuple[PolicyAttempt, ...]
    rules_fallback: bool
    cache_hit: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "parsed": self.parsed.to_dict(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "rules_fallback": self.rules_fallback, "cache_hit": self.cache_hit,
        }


class LLMToolPolicy:
    def __init__(self, llm: BaseLLM, *, max_attempts: int = 2) -> None:
        if not 1 <= max_attempts <= 2:
            raise ValueError("Policy permits at most one repair after the first attempt")
        self.llm, self.max_attempts = llm, max_attempts
        self._cache: dict[str, PolicyDecision] = {}

    def decide(self, text: str) -> PolicyDecision:
        if text in self._cache:
            return replace(self._cache[text], cache_hit=True)
        attempts = []
        request_prompt = (
            POLICY_PROMPT + "\nUSER_REQUEST_JSON=" + json.dumps({"text": text}, ensure_ascii=False)
        )
        prompt = request_prompt
        for number in range(1, self.max_attempts + 1):
            started = perf_counter()
            call, parsed, error = None, None, None
            schema_valid = guard_valid = False
            try:
                call = self.llm.tool_call(prompt, [TOOL_DEFINITION])
                parsed, evidence = validate_policy_call(call)
                schema_valid = True
                guard_policy_decision(text, parsed, evidence)
                guard_valid = True
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            record_method = getattr(self.llm, "generation_record", None)
            generation = record_method() if callable(record_method) else None
            attempts.append(PolicyAttempt(
                number, call, parsed, schema_valid, guard_valid, error, generation,
                (perf_counter() - started) * 1000,
            ))
            if guard_valid:
                result = PolicyDecision(parsed, tuple(attempts), rules_fallback=False)
                break
            observation = observe_budget(text)
            prompt = (
                request_prompt
                + "\nRepair the function call. Previous validation error: " + str(error)
                + "\nBudget observation derived from the request: "
                + json.dumps({
                    "status": observation.status,
                    "price_limit": observation.limit.to_dict() if observation.limit else None,
                    "evidence": observation.evidence,
                }, ensure_ascii=False)
            )
        else:
            result = PolicyDecision(
                parse_recommendation_intent(text), tuple(attempts), rules_fallback=True
            )
        if len(self._cache) >= 128:
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = result
        return result
