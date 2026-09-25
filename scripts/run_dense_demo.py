"""Text query to static item catalog, separate from history-based next-item evaluation."""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.data.storage import load_items
from genrecagent.retrieval.dense import DenseRetriever
from genrecagent.retrieval.encoder import SentenceItemEncoder

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/retrieval.yaml")
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    index = DenseRetriever.load_index(ROOT / config["retrieval"]["index_path"])
    items = load_items(ROOT / config["dataset"]["prepared_path"])
    model_name = index.metadata["model_name"]
    encoder = SentenceItemEncoder(
        model_name,
        cache_dir=ROOT / config["encoder"]["cache_dir"],
        device=config["encoder"]["device"],
        allow_download=args.allow_model_download,
    )
    vector = encoder.encode([args.query], batch_size=1)[0]
    hits = index.search(vector, top_k=args.top_k)
    print(f"Text-query demo: {args.query}; model: {model_name}")
    print(json.dumps(
        [{**hit.to_dict(), "title": items[hit.item_id].title} for hit in hits],
        indent=2,
    ))


if __name__ == "__main__":
    main()
