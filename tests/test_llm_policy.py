import pytest

from genrecagent.agent.constraints import PriceLimit
from genrecagent.agent.policy_guard import guard_policy_decision, observe_budget
from genrecagent.agent.tool_policy import LLMToolPolicy, validate_policy_call
from genrecagent.llm.base import BaseLLM
from genrecagent.llm.tool_json import ToolJSONError, decode_tool_json


def _call(*, intent="personalized", amount=None, inclusive=True, evidence=None):
    return {"name": "recommendation_policy", "arguments": {
        "status": "accepted", "intent": intent, "reason": None,
        "price_limit": None if amount is None else {
            "amount": amount, "inclusive": inclusive, "evidence": evidence,
        },
    }}


class FakeLLM(BaseLLM):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, prompt):
        raise NotImplementedError

    def tool_call(self, prompt, tools):
        assert "USER_REQUEST_JSON" in prompt and len(tools) == 1
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.parametrize("raw", [
    '```json\n{"name":"recommendation_policy"}\n```',
    '{"a":1,"a":2}', '{"x":NaN}', '[{"name":"recommendation_policy"}]',
])
def test_strict_json_rejects_prose_duplicates_and_nonfinite_values(raw) -> None:
    with pytest.raises(ToolJSONError):
        decode_tool_json(raw)


def test_valid_price_plan_and_cached_call() -> None:
    text = "Recommend based on my history at most $20."
    llm = FakeLLM([_call(amount=20, evidence="at most $20")])
    policy = LLMToolPolicy(llm)
    result = policy.decide(text)
    assert result.parsed.price_limit == PriceLimit(20)
    assert result.attempts[0].guard_valid and not result.rules_fallback
    assert policy.decide(text).cache_hit and llm.calls == 1


def test_omitted_budget_is_repaired_without_executing_first_decision() -> None:
    text = "Recommend based on my history under $10."
    policy = LLMToolPolicy(FakeLLM([
        _call(), _call(amount=10, inclusive=False, evidence="under $10"),
    ]))
    result = policy.decide(text)
    assert result.attempts[0].schema_valid and not result.attempts[0].guard_valid
    assert result.parsed.price_limit == PriceLimit(10, False)
    assert len(result.attempts) == 2 and not result.rules_fallback


def test_wrong_boundary_or_provider_failure_has_bounded_rule_fallback() -> None:
    text = "Recommend based on my history under $10."
    wrong = _call(amount=10, inclusive=True, evidence="under $10")
    llm = FakeLLM([wrong, wrong])
    result = LLMToolPolicy(llm).decide(text)
    assert result.rules_fallback and result.parsed.price_limit == PriceLimit(10, False)
    assert llm.calls == 2
    llm = FakeLLM([RuntimeError("offline"), RuntimeError("offline")])
    result = LLMToolPolicy(llm).decide(text)
    assert result.rules_fallback and len(result.attempts) == 2


def test_unknown_tools_or_extra_arguments_and_boolean_amount_are_rejected() -> None:
    call = _call()
    call["name"] = "delete_catalog"
    with pytest.raises(ValueError, match="unknown_tool"):
        validate_policy_call(call)
    call = _call()
    call["arguments"]["secret"] = "unexpected"
    with pytest.raises(ValueError, match="keys"):
        validate_policy_call(call)
    with pytest.raises(ValueError):
        validate_policy_call(_call(amount=True, evidence="at most $1"))


def test_grounded_paraphrase_is_not_limited_to_the_old_intent_grammar() -> None:
    text = "Use my earlier browsing to pick a few products, each no more than 20 dollars."
    result = LLMToolPolicy(FakeLLM([
        _call(amount=20, evidence="no more than 20 dollars"),
    ])).decide(text)
    assert not result.rules_fallback and result.parsed.status == "accepted"


def test_unsupported_conditions_and_unrequested_generative_route_are_blocked() -> None:
    parsed, evidence = validate_policy_call(_call())
    with pytest.raises(ValueError, match="unsupported_condition"):
        guard_policy_decision("Recommend only vegan products.", parsed, evidence)
    parsed, evidence = validate_policy_call(_call(intent="generative"))
    with pytest.raises(ValueError, match="not_explicitly"):
        guard_policy_decision("Use my browsing history to recommend items.", parsed, evidence)


def test_budget_guard_observes_chinese_evidence_and_conflicts() -> None:
    assert observe_budget("参考我的历史，单价最多5美元。").limit == PriceLimit(5)
    assert observe_budget("at most $20 and under $10").status == "invalid_budget"
    assert observe_budget("单价最多20元").status == "unsupported_currency"


def test_multiple_requested_routes_are_not_silently_reduced_to_one() -> None:
    parsed, evidence = validate_policy_call(_call(intent="generative"))
    with pytest.raises(ValueError, match="conflicting_routes"):
        guard_policy_decision("Use popular and generative retrieval.", parsed, evidence)
