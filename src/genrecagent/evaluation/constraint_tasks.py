"""Grade output against task labels and raw metadata, independently of agent filters."""

from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genrecagent.agent.constrained_agent import ConstrainedRecommendation
    from genrecagent.data.models import Item


def grade_constraint_task(
    task: dict, result: "ConstrainedRecommendation", items: dict[str, "Item"],
) -> dict[str, int | bool | str | None]:
    labels = task["expected"]
    budget = labels["price_limit"]
    parsed = result.parsed.to_dict()
    parse_correct = all(parsed[key] == labels[key] for key in ("status", "intent", "price_limit"))
    seen = set(task["history_items"])

    def budget_pass(item_id: str) -> bool:
        if budget is None:
            return True
        value = items[item_id].price
        numeric = isinstance(value, Real) and not isinstance(value, bool)
        if not numeric or not isfinite(float(value)) or value < 0:
            return False
        return value <= budget["amount"] if budget["inclusive"] else value < budget["amount"]

    ids = [hit.item_id for hit in result.hits]
    unknown_price = sum(
        budget is not None and item_id in items and items[item_id].price is None for item_id in ids
    )
    violations = sum(
        item_id not in items or not budget_pass(item_id) for item_id in ids
    )
    safe = (
        len(ids) <= task["top_k"] and len(ids) == len(set(ids)) and violations == 0
        and all(item_id not in seen for item_id in ids)
        and all(hit.rank == rank and isfinite(hit.score)
                for rank, hit in enumerate(result.hits, start=1))
    )
    rejected = labels["status"] != "accepted"
    if rejected:
        contract = (
            parse_correct and result.status == labels["status"] and not ids
            and result.route_trace is None
        )
        return {
            "parse_correct": parse_correct, "contract_pass": contract,
            "rejection_task": True, "rejection_correct": contract,
            "feasible": False, "budget_feasible": False, "fulfilled_feasible": False,
            "price_violations": violations, "unknown_price_admitted": unknown_price,
            "budgeted_returned_items": 0, "expected_status": labels["status"],
            "eligible_count": None,
        }
    eligible_count = sum(
        item_id not in seen and budget_pass(item_id) for item_id in items
    )
    feasible = eligible_count >= task["top_k"]
    expected_status = "fulfilled" if feasible else "insufficient_candidates"
    trace_pass = (
        (result.route_trace is None and eligible_count == 0)
        or (result.route_trace is not None and len(result.route_trace.attempts) <= 3
            and result.route_trace.satisfied
            and all(hit.source == result.route_trace.selected_route for hit in result.hits))
    )
    contract = (
        parse_correct and safe and result.constraints_verified and trace_pass
        and result.status == expected_status and result.eligible_count == eligible_count
        and len(ids) == min(task["top_k"], eligible_count)
    )
    return {
        "parse_correct": parse_correct, "contract_pass": contract,
        "rejection_task": False, "rejection_correct": False,
        "feasible": feasible, "budget_feasible": feasible and budget is not None,
        "fulfilled_feasible": feasible and contract,
        "price_violations": violations, "unknown_price_admitted": unknown_price,
        "budgeted_returned_items": len(ids) if budget is not None else 0,
        "expected_status": expected_status, "eligible_count": eligible_count,
    }
