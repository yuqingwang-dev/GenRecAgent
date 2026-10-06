"""Run the minimal stateful Agent with fixed, prompt-Qwen or SFT-Qwen policy."""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.agent.constraints import PriceLimit
from genrecagent.agent.intent import parse_recommendation_intent
from genrecagent.agent.tool_environment import (
    FixedStepPolicy,
    QwenStepPolicy,
    ToolEnvironment,
    ToolTask,
    run_tool_agent,
)
from genrecagent.agent.tool_policy import LLMToolPolicy
from genrecagent.data.storage import load_items
from genrecagent.llm.local_qwen import LocalQwenLLM
from genrecagent.retrieval.hybrid_loader import load_dual_resources

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["fixed", "prompt", "sft"], default="fixed")
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
    if not torch.cuda.is_available():
        parser.error("CUDA is required for the configured local models")
    if args.policy == "sft" and args.adapter is None:
        parser.error("SFT mode requires a trained adapter")
    if (
        args.policy == "sft"
        and json.loads((args.adapter.parent / "metrics.json").read_text(encoding="utf-8"))["status"]
        != "full_sft"
    ):
        parser.error("A CPU software-smoke adapter is not a full SFT result")
    if args.output and args.output.exists():
        parser.error("Trace output already exists")
    resources = load_dual_resources(ROOT)
    items = load_items(ROOT / "data/processed/All_Beauty_full_optimized")
    policy = FixedStepPolicy()
    parsed = parse_recommendation_intent(args.text)
    if args.policy != "fixed":
        config = yaml.safe_load((ROOT / "configs/llm_policy.yaml").read_text(encoding="utf-8"))
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
        policy = QwenStepPolicy(llm)
        if parsed.status != "accepted":
            if args.policy == "sft":
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
    result = run_tool_agent(ToolEnvironment(resources, items, task), policy)
    result["input_scope"] = "text plus explicit normalized constraints and observed history"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
