import pytest

from genrecagent.retrieval.dense import SearchHit
from genrecagent.retrieval.hybrid import fuse_item_ranks


def test_actual_item_fusion_deduplicates_filters_and_preserves_endpoints() -> None:
    dense = (SearchHit("A", .9, 1, "dense"), SearchHit("B", .8, 2, "dense"))
    gen = (SearchHit("C", -2, 1, "generative"), SearchHit("B", -3, 2, "generative"))
    both = fuse_item_ranks(dense, gen, dense_weight=.5, top_k=3)
    assert [hit.item_id for hit in both] == ["B", "A", "C"]
    assert [hit.rank for hit in both] == [1, 2, 3]
    assert [hit.item_id for hit in fuse_item_ranks(dense, gen, dense_weight=1)] == ["A", "B"]
    assert [hit.item_id for hit in fuse_item_ranks(dense, gen, dense_weight=0)] == ["C", "B"]
    filtered = fuse_item_ranks(dense, gen, seen=frozenset({"B"}), allowed=frozenset({"A"}))
    assert [hit.item_id for hit in filtered] == ["A"]


def test_fusion_rejects_duplicate_source_items_and_bad_settings() -> None:
    bad = (SearchHit("A", 1, 1, "dense"), SearchHit("A", 2, 2, "dense"))
    with pytest.raises(ValueError, match="unique"):
        fuse_item_ranks(bad, ())
    with pytest.raises(ValueError):
        fuse_item_ranks((), (), dense_weight=2)
