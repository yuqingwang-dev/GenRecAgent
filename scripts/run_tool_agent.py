"""Run the minimal stateful Agent with fixed, prompt-Qwen or SFT-Qwen policy."""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.agent.compact_policy import PROTOCOL, CompactQwenStepPolicy
from genrecagent.agent.constraints import PriceLimit
from genrecagent.agent.guarded_policy import run_guarded_agent
from genrecagent.agent.intent import parse_recommendation_intent
from genrecagent.agent.policy_modes import (
    COMPACT_POLICIES,
    GUARDED_POLICIES,
    POLICY_CHOICES,
    SFT_POLICIES,
)
from genrecagent.agent.tool_environment import (
    FixedStepPolicy,
    QwenStepPolicy,
    ToolEnvironment,
    ToolTask,
    run_tool_agent,
)
from genrecagent.agent.tool_policy import LLMToolPolicy
from genrecagent.data.storage import load_items
from genrecagent.delivery.readiness import DEFAULT_ADAPTER, check_artifacts
from genrecagent.evaluation.agent_contracts import end_reason
from genrecagent.llm.local_qwen import LocalQwenLLM
from genrecagent.retrieval.hybrid_loader import load_dual_resources

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=POLICY_CHOICES, default="fixed")
    parser.add_argument("--max-model-rejections", type=int, default=2)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--text", default="Recommend based on my history.")
    parser.add_argument("--history-item", action="append", default=[])
    parser.add_argument(
        "--mode", choices=["dense", "generative", "hybrid", "popularity"], default=None
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-price", type=float)
    parser.add_argument("--category")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    guarded = args.policy in GUARDED_POLICIES
    sft = args.policy in SFT_POLICIES
    compact = args.policy in COMPACT_POLICIES
    if guarded and not 1 <= args.max_model_rejections <= 10:
        parser.error("Model rejection budget must be in [1,10]")
    if args.policy == "guarded_sft" and args.adapter is None:
        args.adapter = ROOT / DEFAULT_ADAPTER
    if args.policy == "compact_sft" and args.adapter is None:
        args.adapter = ROOT / "outputs/tool_sft_recovery_v3_server/tool_sft_recovery_v3/adapter"
    if args.adapter is not None and not sft:
        parser.error("Only an SFT policy may load an adapter")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for the configured local models")
    if sft and args.adapter is None:
        parser.error("SFT mode requires a trained adapter")
    if (
        sft
        and json.loads((args.adapter.parent / "metrics.json").read_text(encoding="utf-8"))["status"]
        != "full_sft"
    ):
        parser.error("A CPU software-smoke adapter is not a full SFT result")
    if sft:
        training = json.loads((args.adapter.parent / "metrics.json").read_text(encoding="utf-8"))
        if compact and training.get("input_protocol") != PROTOCOL:
            parser.error("Compact SFT requires a trained compact-protocol adapter")
        readiness = check_artifacts(ROOT, policy="sft", adapter=args.adapter)
        if not readiness["ready"]:
            parser.error("; ".join(readiness["problems"]))
    if args.output and args.output.exists():
        parser.error("Trace output already exists")
    resources = load_dual_resources(ROOT)
    items = load_items(ROOT / "data/processed/All_Beauty_full_optimized")
    policy = FixedStepPolicy()
    parsed = parse_recommendation_intent(args.text)
    if args.policy != "fixed":
        config = yaml.safe_load((ROOT / "configs/llm_policy.yaml").read_text(encoding="utf-8"))
        if compact:
            config.update(dtype="bfloat16", max_input_tokens=1024, max_new_tokens=96)
        llm = LocalQwenLLM(
            **{
                key: config[key]
                for key in (
                    "model_name",
                    "revision",
                    "device",
                    "dtype",
                    "max_input_tokens",
                    "max_new_tokens",
                )
            },
            cache_dir=ROOT / config["cache_dir"],
            adapter_path=args.adapter,
        )
        policy = CompactQwenStepPolicy(llm) if compact else QwenStepPolicy(llm)
        if parsed.status != "accepted":
            if sft:
                with llm.model.disable_adapter():
                    parsed = LLMToolPolicy(llm).decide(args.text).parsed
            else:
                parsed = LLMToolPolicy(llm).decide(args.text).parsed
    explicit_limit = PriceLimit(args.max_price) if args.max_price is not None else None
    if (
        explicit_limit is not None
        and parsed.price_limit is not None
        and (explicit_limit != parsed.price_limit)
    ):
        parser.error("Explicit price and text price conflict")
    mode = args.mode or {
        "personalized": "dense",
        "popular": "popularity",
        "generative": "generative",
    }.get(parsed.intent, "dense")
    task = ToolTask(
        "cli",
        args.text,
        tuple(args.history_item),
        mode=mode,
        top_k=args.top_k,
        price_limit=parsed.price_limit or explicit_limit,
        category=args.category,
        request_status=parsed.status,
    )
    environment = ToolEnvironment(resources, items, task)
    result = (
        run_guarded_agent(environment, policy, max_model_rejections=args.max_model_rejections)
        if guarded
        else run_tool_agent(environment, policy)
    )
    result["policy"] = args.policy
    result["end_reason"] = end_reason(result)
    result["input_scope"] = "text plus explicit normalized constraints and observed history"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
