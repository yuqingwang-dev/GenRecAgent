"""Inspect one offline history-based routing trace; no LLM or model download."""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.agent.offline_factory import load_offline_router
from genrecagent.agent.offline_router import OfflineRouteRequest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-item", action="append", default=[],
                        help="Observed catalog item ID; repeat in chronological order")
    parser.add_argument("--mode", choices=["auto", "dense", "generative"], default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prepared-dir", type=Path,
                        default=ROOT / "data/processed/All_Beauty_full_optimized")
    parser.add_argument("--index-dir", type=Path,
                        default=ROOT / "data/processed/All_Beauty/index")
    parser.add_argument("--m3-sequences-dir", type=Path,
                        default=ROOT / "data/processed/All_Beauty_m3")
    parser.add_argument("--semantic-ids", type=Path,
                        default=ROOT / "outputs/m2_rqkmeans_full/item_semantic_ids.json")
    parser.add_argument("--generative-config", type=Path,
                        default=ROOT / "configs/generative.yaml")
    args = parser.parse_args()
    enable_generative = args.mode == "generative"
    config = yaml.safe_load(args.generative_config.read_text(encoding="utf-8")) if (
        enable_generative
    ) else None
    resources = load_offline_router(
        args.prepared_dir, args.index_dir,
        enable_generative=enable_generative,
        m3_sequences_dir=args.m3_sequences_dir if enable_generative else None,
        semantic_ids_path=args.semantic_ids if enable_generative else None,
        transition_alpha=float(config["transition"]["alpha"]) if config else 10.0,
        beam_size=int(config["retrieval"]["beam_size"]) if config else 20,
        max_codes=int(config["retrieval"]["max_codes"]) if config else 20,
    )
    result = resources.router.route(OfflineRouteRequest(
        tuple(args.history_item), top_k=args.top_k, mode=args.mode
    ))
    print(json.dumps({
        "history_items": args.history_item,
        "history_policy": "observed_items_only",
        "route_result": result.to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
