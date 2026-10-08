"""Small real-catalog multi-step environment with verified termination."""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from time import perf_counter

from genrecagent.agent.constraints import PriceLimit
from genrecagent.data.models import Item
from genrecagent.retrieval.dense import SearchHit, mean_user_embedding
from genrecagent.retrieval.hybrid import fuse_item_ranks
from genrecagent.retrieval.hybrid_loader import DualResources

TOOL_NAMES = (
    "read_history",
    "retrieve_dense",
    "retrieve_generative",
    "retrieve_popular",
    "filter_candidates",
    "rank_candidates",
    "verify_candidates",
    "finish",
    "clarify",
)
TOOL_SCHEMAS = [
    {
        "name": name,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": (
                {"top_k": {"type": "integer", "minimum": 1, "maximum": 100}}
                if name.startswith("retrieve_")
                else {"reason": {"type": "string"}}
                if name == "clarify"
                else {}
            ),
            "required": ["reason"] if name == "clarify" else [],
        },
    }
    for name in TOOL_NAMES
]
STEP_PROMPT = """Choose one next tool using the actual state. Return JSON name/arguments only.
Read history before retrieval. Attempt all preferred sources. Empty/failed history retrieval can
recover with retrieve_popular. After retrieval call filter_candidates, rank_candidates, then
verify_candidates. Finish only when verification says sufficient and valid. Otherwise recover
with another source or clarify without weakening price/category constraints. No repeated tools
on unchanged data. Rejected requests and an empty eligible catalog need clarification.
"""


@dataclass(frozen=True, slots=True)
class ToolTask:
    task_id: str
    text: str
    history: tuple[str, ...]
    mode: str = "hybrid"
    top_k: int = 5
    price_limit: PriceLimit | None = None
    category: str | None = None
    request_status: str = "accepted"
    injected_failures: frozenset[str] = frozenset()


@dataclass(slots=True)
class ToolEnvironment:
    resources: DualResources
    items: dict[str, Item]
    task: ToolTask
    max_steps: int = 10
    max_seconds: float = 120.0
    trace: list[dict] = field(default_factory=list)
    sources: dict[str, tuple[SearchHit, ...]] = field(default_factory=dict)
    source_errors: dict[str, str] = field(default_factory=dict)
    history_read: bool = False
    filtered: bool = False
    ranked: tuple[SearchHit, ...] = ()
    ranked_ready: bool = False
    verified: bool = False
    sufficient: bool = False
    status: str | None = None
    epoch: int = 0
    allowed: frozenset[str] = field(init=False)
    eligible_count: int = field(init=False)
    _repeats: set[str] = field(default_factory=set)
    _started: float = field(default_factory=perf_counter)

    def __post_init__(self) -> None:
        if self.task.mode not in {"dense", "generative", "hybrid", "popularity"}:
            raise ValueError("Unknown route mode")
        if not 1 <= self.task.top_k <= 100 or self.max_steps < 1:
            raise ValueError("Invalid request or execution budget")
        if set(self.items) != set(self.resources.dual.dense.item_ids):
            raise ValueError("Metadata and retrieval catalogs differ")
        self.allowed = frozenset(
            item_id
            for item_id, item in self.items.items()
            if (self.task.price_limit is None or self.task.price_limit.allows(item.price))
            and (self.task.category is None or self.task.category in item.categories)
        )
        self.eligible_count = len(self.allowed - set(self.task.history))

    @property
    def required_sources(self) -> set[str]:
        return (
            {"dense", "generative"}
            if self.task.mode == "hybrid"
            else {"popularity" if self.task.mode == "popularity" else self.task.mode}
        )

    def snapshot(self) -> dict:
        return {
            "request": {
                "text": self.task.text,
                "mode": self.task.mode,
                "top_k": self.task.top_k,
                "price_limit": (self.task.price_limit.to_dict() if self.task.price_limit else None),
                "category": self.task.category,
                "status": self.task.request_status,
            },
            "history_read": self.history_read,
            "observed_history": list(self.task.history[:5]) if self.history_read else None,
            "source_counts": {name: len(rows) for name, rows in self.sources.items()},
            "source_errors": dict(self.source_errors),
            "pending_sources": sorted(
                self.required_sources - (self.sources.keys() | self.source_errors.keys())
            ),
            "eligible_count": self.eligible_count,
            "filtered": self.filtered,
            "ranked_count": len(self.ranked),
            "ranked_ready": self.ranked_ready,
            "verified": self.verified,
            "sufficient": self.sufficient,
            "steps_left": self.max_steps - len(self.trace),
            "last_observation": deepcopy(self.trace[-1]["observation"]) if self.trace else None,
        }

    def step(self, call: dict) -> dict:
        if self.status is not None:
            raise ValueError("Environment already terminated")
        if len(self.trace) >= self.max_steps or perf_counter() - self._started > self.max_seconds:
            self.status = "budget_exhausted"
            return {"ok": False, "error": "execution_budget_exhausted"}
        before, started = self.snapshot(), perf_counter()
        name, args = call.get("name"), call.get("arguments")
        observation = {}
        service_failure = False
        try:
            if (
                set(call) != {"name", "arguments"}
                or not isinstance(name, str)
                or name not in TOOL_NAMES
                or not isinstance(args, dict)
            ):
                raise ValueError("Unknown tool or malformed call")
            expected = (
                {"top_k"}
                if name.startswith("retrieve_")
                else {"reason"}
                if name == "clarify"
                else set()
            )
            if set(args) - expected:
                raise ValueError("Unexpected arguments")
            signature = json.dumps(call, sort_keys=True) + str(
                self.epoch
                if name in {"filter_candidates", "rank_candidates", "verify_candidates"}
                else self.history_read
                if name.startswith("retrieve_")
                else self.verified
                if name == "finish"
                else ""
            )
            if signature in self._repeats:
                raise ValueError("Repeated action on unchanged observations")
            self._repeats.add(signature)
            observation = self._execute(name, args)
            observation["ok"] = True
        except Exception as exc:
            observation = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if (
                isinstance(name, str)
                and name in {"retrieve_dense", "retrieve_generative", "retrieve_popular"}
                and (self.history_read and not isinstance(exc, ValueError))
            ):
                key = {
                    "retrieve_dense": "dense",
                    "retrieve_generative": "generative",
                    "retrieve_popular": "popularity",
                }[name]
                self.source_errors[key] = observation["error"]
                self.epoch += 1
                service_failure = True
        self.trace.append(
            {
                "state": before,
                "action": deepcopy(call),
                "observation": observation,
                "latency_ms": (perf_counter() - started) * 1000,
            }
        )
        if service_failure:
            self.trace[-1]["service_failure"] = True
        # Artifact-only audit. Never place full candidate pools into the model's observation.
        if observation.get("ok") and name.startswith("retrieve_"):
            key = observation["source"]
            self.trace[-1]["candidate_audit"] = {
                "kind": "retrieved_pool",
                "source": key,
                "hits": [hit.to_dict() for hit in self.sources[key]],
            }
        elif observation.get("ok") and name == "filter_candidates":
            self.trace[-1]["candidate_audit"] = {
                "kind": "filtered_pools",
                "sources": {
                    key: [hit.to_dict() for hit in rows] for key, rows in self.sources.items()
                },
            }
        return observation

    def _execute(self, name: str, args: dict) -> dict:
        if self.task.request_status != "accepted" and name != "clarify":
            raise ValueError("Rejected requests only permit clarification")
        if name == "read_history":
            if self.history_read:
                raise ValueError("History already read")
            self.history_read = True
            return {"history_items": list(self.task.history)}
        if name.startswith("retrieve_"):
            if not self.history_read:
                raise ValueError("Read history first")
            count = args.get("top_k", 50)
            if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 100:
                raise ValueError("top_k must be an integer in [1,100]")
            if name in self.task.injected_failures:
                raise RuntimeError("synthetic_injected_tool_failure")
            seen = set(self.task.history)
            if name == "retrieve_dense":
                vector = mean_user_embedding(self.task.history, self.resources.dual.dense)
                rows = (
                    ()
                    if vector is None
                    else tuple(
                        self.resources.dual.dense.search(
                            vector, top_k=count, exclude_seen_items=seen
                        )
                    )
                )
                key = "dense"
            elif name == "retrieve_generative":
                rows = self.resources.dual.generative.retrieve(self.task.history, top_k=count).hits
                key = "generative"
            else:
                rows = tuple(
                    self.resources.base.router.popularity.search(
                        top_k=count, exclude_seen_items=seen, allowed_item_ids=self.allowed
                    )
                )
                key = "popularity"
            self.sources[key] = rows
            self.filtered, self.verified, self.sufficient = False, False, False
            self.ranked = ()
            self.ranked_ready = False
            self.epoch += 1
            return {"source": key, "count": len(rows), "items": [h.to_dict() for h in rows[:3]]}
        if name == "filter_candidates":
            if self.eligible_count and self.required_sources - (
                self.sources.keys() | self.source_errors.keys()
            ):
                raise ValueError("Required sources have not been attempted")
            self.sources = {
                source: tuple(
                    SearchHit(h.item_id, h.score, rank, h.source)
                    for rank, h in enumerate(
                        (
                            h
                            for h in rows
                            if h.item_id in self.allowed and h.item_id not in self.task.history
                        ),
                        1,
                    )
                )
                for source, rows in self.sources.items()
            }
            self.filtered = True
            return {
                "eligible_count": self.eligible_count,
                "source_counts": {key: len(rows) for key, rows in self.sources.items()},
            }
        if name == "rank_candidates":
            if not self.filtered:
                raise ValueError("Filter candidates before ranking")
            dense, gen = self.sources.get("dense", ()), self.sources.get("generative", ())
            weight = 0.75 if dense and gen else 1.0 if dense else 0.0
            ranked = list(fuse_item_ranks(dense, gen, dense_weight=weight, top_k=self.task.top_k))
            ids = {hit.item_id for hit in ranked}
            for hit in self.sources.get("popularity", ()):
                if hit.item_id not in ids and len(ranked) < self.task.top_k:
                    ids.add(hit.item_id)
                    ranked.append(SearchHit(hit.item_id, hit.score, len(ranked) + 1, "popularity"))
            self.ranked = tuple(ranked)
            self.ranked_ready = True
            return {"count": len(ranked), "items": [h.to_dict() for h in ranked[:3]]}
        if name == "verify_candidates":
            if not self.filtered or not self.ranked_ready:
                raise ValueError("Filter and rank before verification")
            pool = {hit.item_id for rows in self.sources.values() for hit in rows}
            self.verified = len({h.item_id for h in self.ranked}) == len(self.ranked) and all(
                h.item_id in pool
                and h.item_id in self.allowed
                and h.item_id not in self.task.history
                and isfinite(h.score)
                and h.rank == rank
                for rank, h in enumerate(self.ranked, 1)
            )
            self.sufficient = self.verified and len(self.ranked) == self.task.top_k
            return {
                "valid": self.verified,
                "sufficient": self.sufficient,
                "count": len(self.ranked),
            }
        if name == "finish":
            if not self.verified or not self.sufficient:
                raise ValueError("Verified sufficient candidates required before finish")
            self.status = "success"
            return {"status": self.status, "items": [h.to_dict() for h in self.ranked]}
        if name == "clarify":
            if not isinstance(args.get("reason"), str) or not args["reason"].strip():
                raise ValueError("Clarification reason required")
            self.status = "clarification"
            return {"status": self.status, "reason": args["reason"]}
        raise ValueError("Unsupported action")

    def result(self) -> dict:
        if self.status is None:
            self.status = "budget_exhausted"
        return {
            "task_id": self.task.task_id,
            "status": self.status,
            "items": [
                dict(
                    h.to_dict(),
                    metadata_price=self.items[h.item_id].price,
                    categories=list(self.items[h.item_id].categories),
                )
                for h in self.ranked
            ]
            if self.status == "success"
            else [],
            "steps": len(self.trace),
            "trace": self.trace,
            "invalid_calls": sum(not row["observation"]["ok"] for row in self.trace),
        }


class FixedStepPolicy:
    def choose(self, state: dict) -> dict:
        def action(name, **arguments):
            return {"name": name, "arguments": arguments}

        if state["request"]["status"] != "accepted":
            return action("clarify", reason="request_invalid_or_unsupported")
        if state["eligible_count"] == 0:
            return (
                action("clarify", reason="no_eligible_catalog_items")
                if state["filtered"]
                else (action("filter_candidates"))
            )
        if not state["history_read"]:
            return action("read_history")
        if state["pending_sources"]:
            source = state["pending_sources"][0]
            return action(
                {
                    "dense": "retrieve_dense",
                    "generative": "retrieve_generative",
                    "popularity": "retrieve_popular",
                }[source],
                top_k=50,
            )
        if sum(state["source_counts"].values()) < state["request"]["top_k"] and (
            "popularity" not in state["source_counts"]
            and "popularity" not in state["source_errors"]
        ):
            return action("retrieve_popular", top_k=50)
        if not state["filtered"]:
            return action("filter_candidates")
        if not state["ranked_ready"]:
            return action("rank_candidates")
        if not state["verified"]:
            return action("verify_candidates")
        if state["sufficient"]:
            return action("finish")
        if "popularity" not in state["source_counts"]:
            return action("retrieve_popular", top_k=100)
        return action("clarify", reason="insufficient_candidates_under_constraints")


class QwenStepPolicy:
    def __init__(self, llm) -> None:
        self.llm = llm

    def choose(self, state: dict) -> dict:
        return self.llm.tool_call(
            STEP_PROMPT + "\nSTATE=" + json.dumps(state, ensure_ascii=False), TOOL_SCHEMAS
        )


def run_tool_agent(environment: ToolEnvironment, policy, *, recover: bool = True) -> dict:
    while environment.status is None and len(environment.trace) < environment.max_steps:
        if not recover and environment.trace and not environment.trace[-1]["observation"]["ok"]:
            environment.status = "tool_failure"
            break
        try:
            call = policy.choose(environment.snapshot())
        except Exception as exc:
            call = {"name": "policy_error", "arguments": {"error": type(exc).__name__}}
        previous_steps = len(environment.trace)
        environment.step(call)
        if isinstance(policy, QwenStepPolicy) and len(environment.trace) > previous_steps:
            environment.trace[-1]["policy_generation"] = policy.llm.generation_record()
    return environment.result()
