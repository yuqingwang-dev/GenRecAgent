import pytest

from genrecagent.evaluation.diagnostics import RetrievalBucket, validation_cohorts
from genrecagent.retrieval.dense import SearchHit
from genrecagent.retrieval.generative import CandidateDiagnostics, GenerativeSearchResult
from genrecagent.retrieval.semantic_trie import CodePrediction

TARGET = ("a", "b", "c")
OTHER = ("d", "e", "f")


def _result(*, target_code_found: bool, item_rank: int | None) -> GenerativeSearchResult:
    codes = (CodePrediction(TARGET if target_code_found else OTHER, -1.0, 1),)
    hits = () if item_rank is None else (
        SearchHit("target", -1.0, item_rank, "generative"),
    )
    return GenerativeSearchResult(
        hits=hits, predicted_codes=codes, hit_codes=(),
        diagnostics=CandidateDiagnostics(1, 0, 0, 0, len(hits), 0, 0, 0),
    )


def test_bucket_partitions_code_and_item_misses() -> None:
    bucket = RetrievalBucket()
    bucket.add("target", TARGET, _result(target_code_found=False, item_rank=None))
    bucket.add("target", TARGET, _result(target_code_found=True, item_rank=None))
    bucket.add("target", TARGET, _result(target_code_found=True, item_rank=2))
    summary = bucket.summary()
    assert summary["users"] == 3
    assert summary["code_not_predicted"] == 1
    assert summary["code_predicted_item_missed"] == 1
    assert summary["item_hit"] == 1
    assert summary["Recall@5"] == pytest.approx(1 / 3)
    assert summary["CodeRecall@20"] == pytest.approx(2 / 3)
    assert summary["NDCG@20"] == pytest.approx(1 / (3 * 1.584962500721156))


def test_validation_cohorts_are_disjoint_and_validate_inputs() -> None:
    assert validation_cohorts(history_length=1, target_code_seen=False, collision_size=5) == (
        ("all", "all"), ("history_length", "1"),
        ("target_code_seen_in_train", "no"), ("target_collision_size", "5_plus"),
    )
    assert validation_cohorts(history_length=2, target_code_seen=True, collision_size=2)[1:] == (
        ("history_length", "2"), ("target_code_seen_in_train", "yes"),
        ("target_collision_size", "2_to_4"),
    )
    with pytest.raises(ValueError):
        validation_cohorts(history_length=0, target_code_seen=True, collision_size=1)
