import numpy as np
import pytest

from genrecagent.data.models import UserHistory
from genrecagent.evaluation.baseline import evaluate_baselines
from genrecagent.evaluation.recommendation import ndcg_at_k, recall_at_k
from genrecagent.retrieval.dense import DenseRetriever, mean_user_embedding
from genrecagent.retrieval.encoder import normalize_rows
from genrecagent.retrieval.popularity import PopularityRetriever


def _history() -> UserHistory:
    return UserHistory("u", ("A", "B"), "C", "D", (10, 20), 30, 40)


def _index() -> DenseRetriever:
    index = DenseRetriever()
    index.build_index(
        ["A", "B", "C", "D", "E"],
        np.array([[1, 0], [0, 1], [1, 1], [1, 0.1], [0, 0.5]], dtype=np.float32),
    )
    return index


def test_embedding_normalization_and_exact_mapping() -> None:
    normalized = normalize_rows(np.array([[3, 4], [0, 2]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1)
    index = _index()
    user_vector = mean_user_embedding(("A", "B"), index)
    assert user_vector is not None
    assert np.linalg.norm(user_vector) == pytest.approx(1)
    hits = index.search(user_vector, top_k=3, exclude_seen_items={"A", "B"})
    assert hits[0].item_id == "C"
    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert all(hit.item_id not in {"A", "B"} for hit in hits)


def test_save_load_preserves_item_mapping_and_results(tmp_path) -> None:
    index = _index()
    query = np.array([1, 0], dtype=np.float32)
    before = index.search(query, top_k=3, exclude_seen_items={"A"})
    path = tmp_path / "index"
    index.save_index(path)
    loaded = DenseRetriever.load_index(path)
    after = loaded.search(query, top_k=3, exclude_seen_items={"A"})
    assert [hit.item_id for hit in before] == [hit.item_id for hit in after]
    assert [hit.score for hit in before] == pytest.approx([hit.score for hit in after])
    assert loaded.metadata["normalization"] == "l2"


def test_popularity_uses_train_only_and_evaluation_histories() -> None:
    history = _history()
    popularity = PopularityRetriever([history], ["A", "B", "C", "D", "E"])
    assert popularity.counts["C"] == 0
    assert popularity.counts["D"] == 0
    assert popularity.search(top_k=2, exclude_seen_items={"A"})[0].item_id == "B"
    index = _index()
    validation = evaluate_baselines([history], index, split="validation", k_values=(1, 5))
    test = evaluate_baselines([history], index, split="test", k_values=(1, 5))
    assert validation["history_policy"] == "train"
    assert test["history_policy"] == "train_plus_validation"
    assert validation["methods"]["dense"]["Recall@1"] == 1.0
    assert test["methods"]["dense"]["Recall@5"] == 1.0


def test_single_target_metrics_are_one_based_and_unreachable_not_dropped() -> None:
    assert recall_at_k(["A", "B"], {"B"}, 1) == 0
    assert recall_at_k(["A", "B"], {"B"}, 2) == 1
    assert ndcg_at_k(["A", "B"], {"B"}, 2) == pytest.approx(1 / np.log2(3))
    history = UserHistory("u", ("A",), "Z", "Y", (10,), 20, 30)
    result = evaluate_baselines([history], _index(), split="validation", k_values=(5,))
    assert result["users"] == 1
    assert result["unreachable_targets"] == 1
    assert result["methods"]["dense"]["Recall@5"] == 0
