"""Aggregate independently graded constrained recommendation task records."""

from collections import Counter
from statistics import mean


def summarize_task_records(records: list[dict]) -> dict[str, object]:
    grades = [row["grade"] for row in records]
    return {
        "tasks": len(records),
        "parse_correct": sum(g["parse_correct"] for g in grades),
        "contract_pass": sum(g["contract_pass"] for g in grades),
        "feasible_tasks": sum(g["feasible"] for g in grades),
        "feasible_fulfilled": sum(g["fulfilled_feasible"] for g in grades),
        "budget_feasible_tasks": sum(g["budget_feasible"] for g in grades),
        "budget_feasible_fulfilled": sum(
            g["budget_feasible"] and g["fulfilled_feasible"] for g in grades
        ),
        "rejection_tasks": sum(g["rejection_task"] for g in grades),
        "rejections_correct": sum(g["rejection_correct"] for g in grades),
        "budgeted_returned_items": sum(g["budgeted_returned_items"] for g in grades),
        "price_violations": sum(g["price_violations"] for g in grades),
        "unknown_price_admitted": sum(g["unknown_price_admitted"] for g in grades),
        "fallback_tasks": sum(
            bool(row["result"]["route_trace"] and row["result"]["route_trace"]["fallback_used"])
            for row in records
        ),
        "outcomes": dict(Counter(row["result"]["status"] for row in records)),
        "avg_request_latency_ms": mean(row["latency_ms"] for row in records),
    }
