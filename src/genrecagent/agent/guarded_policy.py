"""State-gated model execution with explicit, counted controller intervention."""

from copy import deepcopy
from time import perf_counter

from genrecagent.agent.tool_environment import TOOL_NAMES, FixedStepPolicy


def allowed_actions(state: dict) -> tuple[str, ...]:
    """Compute prerequisites from observable state, not expected task labels."""
    request = state["request"]
    if request["status"] != "accepted" or state["eligible_count"] < request["top_k"]:
        return ("clarify",)
    if not state["history_read"]:
        return ("read_history",)
    if state["pending_sources"]:
        names = {
            "dense": "retrieve_dense",
            "generative": "retrieve_generative",
            "popularity": "retrieve_popular",
        }
        return tuple(names[source] for source in state["pending_sources"])
    popular_available = (
        "popularity" not in state["source_counts"] and "popularity" not in state["source_errors"]
    )
    count = sum(state["source_counts"].values())
    if count < request["top_k"] and popular_available:
        return ("retrieve_popular",)
    if not state["filtered"]:
        return ("filter_candidates",)
    if not state["ranked_ready"]:
        return ("rank_candidates",)
    if not state["verified"]:
        last = state.get("last_observation") or {}
        if last.get("ok") and last.get("valid") is False:
            return ("retrieve_popular",) if popular_available else ("clarify",)
        return ("verify_candidates",)
    if state["sufficient"]:
        return ("finish",)
    return ("retrieve_popular",) if popular_available else ("clarify",)


def rejection_reason(call: object, state: dict, allowed: tuple[str, ...]) -> str | None:
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        return "malformed_call"
    name, args = call["name"], call["arguments"]
    if name not in TOOL_NAMES or not isinstance(args, dict):
        return "unknown_tool_or_arguments"
    expected = (
        {"top_k"} if name.startswith("retrieve_") else {"reason"} if name == "clarify" else set()
    )
    if set(args) - expected:
        return "unexpected_arguments"
    if name.startswith("retrieve_"):
        count = args.get("top_k", 50)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not state["request"]["top_k"] <= count <= 100
        ):
            return "retrieval_count_out_of_contract"
    if name == "clarify" and (
        not isinstance(args.get("reason"), str) or not args["reason"].strip()
    ):
        return "missing_clarification_reason"
    return None if name in allowed else "action_precondition_or_premature_termination"


def controller_action(state: dict, allowed: tuple[str, ...]) -> dict:
    preferred = FixedStepPolicy().choose(state)
    if rejection_reason(preferred, state, allowed) is None:
        return preferred
    name = allowed[0]
    arguments = {}
    if name.startswith("retrieve_"):
        arguments = {"top_k": min(100, max(50, state["request"]["top_k"]))}
    elif name == "clarify":
        arguments = {"reason": "catalog_infeasible_or_execution_cannot_safely_complete"}
    return {"name": name, "arguments": arguments}


def run_guarded_agent(environment, policy, *, max_model_rejections: int = 2) -> dict:
    if not 1 <= max_model_rejections <= 10:
        raise ValueError("Model rejection budget must be in [1,10]")
    counts = {
        "model_proposals": 0,
        "model_accepted_actions": 0,
        "model_rejected_proposals": 0,
        "controller_actions": 0,
        "handoff_to_controller": False,
        "model_generation_tokens": 0,
    }
    unexecuted_proposals = []
    while environment.status is None and len(environment.trace) < environment.max_steps:
        if perf_counter() - environment._started > environment.max_seconds:
            break
        state = environment.snapshot()
        allowed = allowed_actions(state)
        # Processing invariant failures must not loop or permit a forged successful finish.
        if environment.trace:
            previous = environment.trace[-1]
            if not previous["observation"]["ok"] and previous["action"]["name"] in {
                "filter_candidates",
                "rank_candidates",
                "verify_candidates",
            }:
                allowed = ("clarify",)
        terminal_rule = (
            state["request"]["status"] != "accepted"
            or state["eligible_count"] < state["request"]["top_k"]
        )
        proposal, rejected, generation = None, None, None
        source = "controller"
        if not counts["handoff_to_controller"] and not terminal_rule:
            counts["model_proposals"] += 1
            try:
                proposal = policy.choose(state)
                rejected = rejection_reason(proposal, state, allowed)
            except Exception as exc:
                rejected = f"policy_exception:{type(exc).__name__}"
            llm = getattr(policy, "llm", None)
            if llm is not None:
                generation = llm.generation_record()
                if generation is not None:
                    counts["model_generation_tokens"] += generation.get("output_tokens", 0)
            if rejected is None:
                source = "model"
            else:
                counts["model_rejected_proposals"] += 1
                if counts["model_rejected_proposals"] >= max_model_rejections:
                    counts["handoff_to_controller"] = True
        call = proposal if source == "model" else controller_action(state, allowed)
        previous_steps = len(environment.trace)
        environment.step(call)
        if len(environment.trace) == previous_steps:
            if proposal is not None or generation is not None:
                unexecuted_proposals.append(
                    {
                        "model_proposal": deepcopy(proposal),
                        "generation": generation,
                        "rejection_reason": rejected,
                        "reason": "deadline_before_tool_execution",
                    }
                )
            break
        counts["model_accepted_actions" if source == "model" else "controller_actions"] += 1
        environment.trace[-1]["guard_decision"] = {
            "source": source,
            "allowed_actions": list(allowed),
            "model_proposal": deepcopy(proposal),
            "rejection_reason": rejected,
            "terminal_catalog_or_request_rule": terminal_rule,
            "handoff_to_controller": counts["handoff_to_controller"],
        }
        if generation is not None:
            # Includes blocked model generation, so latency/tokens are never hidden.
            environment.trace[-1]["policy_generation"] = generation
    result = environment.result()
    result["guard"] = {
        **counts,
        "max_model_rejections": max_model_rejections,
        "unexecuted_proposals": unexecuted_proposals,
        "scope": "guarded system; controller success is not raw model success",
    }
    return result
