"""Run one controlled-language recommendation with hard metadata price checks."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.agent.constrained_agent import ConstrainedRecommendationAgent
from genrecagent.agent.intent import parse_recommendation_intent
from genrecagent.agent.offline_factory import load_offline_router
from genrecagent.data.storage import load_items

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--history-item", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prepared-dir", type=Path,
                        default=ROOT / "data/processed/All_Beauty_full_optimized")
    parser.add_argument("--index-dir", type=Path,
                        default=ROOT / "data/processed/All_Beauty/index")
    args = parser.parse_args()
    parsed = parse_recommendation_intent(args.text)
    resources = load_offline_router(
        args.prepared_dir, args.index_dir, enable_generative=parsed.intent == "generative",
        m3_sequences_dir=ROOT / "data/processed/All_Beauty_m3",
        semantic_ids_path=ROOT / "outputs/m2_rqkmeans_full/item_semantic_ids.json",
    )
    items = load_items(args.prepared_dir)
    result = ConstrainedRecommendationAgent(resources.router, items).run(
        args.text, tuple(args.history_item), top_k=args.top_k,
    )
    payload = result.to_dict()
    for item in payload["items"]:
        item["title"] = items[item["item_id"]].title
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
