"""Local CPU fixture checks the CLI data boundary and baseline wiring."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.build_index import build_catalog_index

from genrecagent.data.storage import load_histories
from genrecagent.evaluation.baseline import evaluate_baselines
from genrecagent.retrieval.dense import DenseRetriever


class FakeEncoder:
    model_name = "fake-fixture-encoder"
    device = "cpu"

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        assert batch_size > 0
        vectors = {
            "A": [1, 0],
            "B": [0, 1],
            "C": [1, 1],
            "D": [0.8, 0.2],
        }
        return np.asarray(
            [vectors[text.split("Product ")[1][0]] for text in texts], dtype=np.float32
        )


def test_local_parquet_prepare_and_cpu_baselines(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    metadata_file = tmp_path / "metadata.parquet"
    review_file = tmp_path / "reviews.parquet"
    metadata = [
        {
            "parent_asin": key,
            "title": f"Product {key}",
            "description": [f"Description {key}"],
            "features": ["beauty"],
            "categories": ["All Beauty"],
            "main_category": "All Beauty",
            "price": 9.0,
            "average_rating": 4.0,
        }
        for key in ("A", "B", "C", "D")
    ]
    reviews = [
        {
            "parent_asin": key,
            "user_id": user,
            "rating": 4.0,
            "timestamp": time,
            "verified_purchase": False,
        }
        for user in ("u1", "u2")
        for key, time in (("A", 10), ("B", 20), ("C", 30), ("D", 40))
    ]
    pq.write_table(pa.Table.from_pylist(metadata), metadata_file)
    pq.write_table(pa.Table.from_pylist(reviews), review_file)
    output = tmp_path / "prepared"
    run = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/prepare_amazon.py"),
            "--review-file", str(review_file),
            "--metadata-file", str(metadata_file),
            "--source-kind", "synthetic_fixture",
            "--output-dir", str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"users": 2' in run.stdout
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_kind"] == "synthetic_fixture"
    assert manifest["stats"]["users"] == 2
    assert manifest["stats"]["train_interactions"] == 4
    histories = load_histories(output)
    index = build_catalog_index(
        output,
        FakeEncoder(),
        batch_size=2,
        backend="numpy",
        max_items=None,
        seed=42,
        dataset_manifest=manifest,
    )
    index.save_index(tmp_path / "index")
    index = DenseRetriever.load_index(tmp_path / "index")
    result = evaluate_baselines(histories, index, split="validation", k_values=(5,))
    assert result["users"] == 2
    assert result["methods"]["dense"]["Recall@5"] == 1.0
    evaluation_dir = tmp_path / "evaluation"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/evaluate_dense.py"),
            "--prepared-dir", str(output),
            "--index-dir", str(tmp_path / "index"),
            "--output-dir", str(evaluation_dir),
            "--split", "validation",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    saved = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["validation"]["users"] == 2
    assert saved["validation"]["methods"]["dense"]["Recall@5"] == 1.0
