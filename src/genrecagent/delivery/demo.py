"""Independently grade demo products; never count empty success as fulfillment."""

from math import isfinite


def grade_demo(result: dict, items: dict, *, history, top_k: int, max_price: float) -> dict:
    eligible = {
        item_id
        for item_id, item in items.items()
        if item_id not in history
        and item.price is not None
        and isfinite(item.price)
        and 0 <= item.price <= max_price
    }
    rows = result["items"]
    ids = [row["item_id"] for row in rows]
    safe = len(ids) == len(set(ids)) and all(item_id in eligible for item_id in ids)
    expected = "success" if len(eligible) >= top_k else "clarification"
    correct = (
        safe
        and result["status"] == expected
        and (len(rows) == top_k if expected == "success" else not rows)
    )
    return {
        "contract_pass": bool(correct),
        "expected_status": expected,
        "eligible_catalog_items": len(eligible),
        "returned_items": len(rows),
        "safe_items": safe,
    }
