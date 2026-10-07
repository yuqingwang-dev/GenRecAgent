"""Deterministic, action-balanced selection; all inputs are executed training examples."""

import random
from collections import Counter, defaultdict

from genrecagent.agent.tool_environment import TOOL_NAMES

TRAIN_TEMPLATES = (
    "Recommend with {mode}; each product costs at most ${price}.",
    "Use {mode} and return {top_k} unseen items under an inclusive ${price} cap.",
    "按{mode}路径推荐{top_k}件商品，单价至多{price}美元。",
)
VALIDATION_TEMPLATES = (
    "For this request select {top_k} unseen products via {mode}, budget ceiling ${price} each.",
    "本次采用{mode}，给出{top_k}个未看过的商品，每件不高于{price}美元。",
    "Route requested: {mode}. Item count: {top_k}. Maximum unit price: {price} USD.",
)
TEST_TEMPLATES = (
    "My unit budget is USD {price}; please choose {top_k} unseen products through {mode}.",
    "请走{mode}推荐路线，排除已看商品，选{top_k}件且每件最多{price}美元。",
    "Using mode={mode}, supply {top_k} new candidates "
    "with a per-item inclusive cap of {price} USD.",
)


def balanced_actions(
    rows: list[dict], *, per_action: int = 120, seed: int = 42
) -> tuple[list, dict]:
    if per_action < 1:
        raise ValueError("Positive action sample count required")
    buckets = defaultdict(list)
    for row in rows:
        if row["split"] != "train" or not row["executed_action_checked"]:
            raise ValueError("Only checked training actions may be sampled")
        buckets[row["action"]["name"]].append(row)
    if set(buckets) != set(TOOL_NAMES):
        raise ValueError("Every physical/terminal tool needs actual training coverage")
    rng = random.Random(seed)
    selected = []
    for name in TOOL_NAMES:
        candidates = list(buckets[name])
        rng.shuffle(candidates)
        chosen = candidates[:per_action]
        chosen += [rng.choice(candidates) for _ in range(per_action - len(chosen))]
        selected.extend(chosen)
    rng.shuffle(selected)
    return selected, {
        "candidate_actions": dict(Counter(row["action"]["name"] for row in rows)),
        "selected_actions": dict(Counter(row["action"]["name"] for row in selected)),
        "selected_rows": len(selected),
        "distinct_selected_actions": len({row["sample_id"] for row in selected}),
        "oversampling": "replacement only within train action buckets; "
        "duplicates are not new trajectories",
    }


def check_split_groups(records: dict[str, list[dict]]) -> None:
    users = {split: {row["user_group"] for row in rows} for split, rows in records.items()}
    goals = {
        split: {(row["mode"], row["max_price"], row["category"], row["top_k"]) for row in rows}
        for split, rows in records.items()
    }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if users[a] & users[b] or goals[a] & goals[b]:
            raise ValueError("User or normalized-goal leakage across recovery splits")
    if set(TRAIN_TEMPLATES) & (set(VALIDATION_TEMPLATES) | set(TEST_TEMPLATES)) or set(
        VALIDATION_TEMPLATES
    ) & set(TEST_TEMPLATES):
        raise ValueError("Template leakage")
