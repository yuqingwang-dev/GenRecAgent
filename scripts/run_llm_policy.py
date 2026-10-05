"""Inspect a pinned local Qwen tool decision and optionally execute its verified plan."""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.agent.constrained_agent import ConstrainedRecommendationAgent
from genrecagent.agent.offline_factory import load_offline_router
from genrecagent.agent.tool_policy import LLMToolPolicy
from genrecagent.data.storage import load_items
from genrecagent.llm.local_qwen import LocalQwenLLM

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--history-item", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/llm_policy.yaml")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    llm = LocalQwenLLM(
        **{key: config[key] for key in (
            "model_name", "revision", "device", "dtype", "max_input_tokens", "max_new_tokens"
        )}, cache_dir=ROOT / config["cache_dir"], allow_download=False,
    )
    decision = LLMToolPolicy(llm).decide(args.text)
    recommendation = None
    if not args.plan_only:
        prepared = ROOT / "data/processed/All_Beauty_full_optimized"
        resources = load_offline_router(
            prepared, ROOT / "data/processed/All_Beauty/index",
            enable_generative=decision.parsed.intent == "generative",
            m3_sequences_dir=ROOT / "data/processed/All_Beauty_m3",
            semantic_ids_path=ROOT / "outputs/m2_rqkmeans_full/item_semantic_ids.json",
        )
        recommendation = ConstrainedRecommendationAgent(
            resources.router, load_items(prepared)
        ).execute_intent(decision.parsed, tuple(args.history_item), top_k=args.top_k).to_dict()
    print(json.dumps({
        "model": llm.runtime, "policy": decision.to_dict(), "recommendation": recommendation,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
