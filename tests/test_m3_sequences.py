import pytest

from genrecagent.data.models import UserHistory
from genrecagent.data.sequences import build_sequence_examples
from genrecagent.semantic_id.codebook import SemanticCodebook


def _mapping() -> SemanticCodebook:
    return SemanticCodebook(assignments={
        item: (f"<s0:{index}>", "<s1:0>", "<s2:0>")
        for index, item in enumerate("ABCDE")
    })


def test_train_prefixes_never_use_validation_or_test_targets() -> None:
    history = UserHistory("u", ("A", "B", "C"), "D", "E", (10, 20, 30), 40, 50)
    examples = build_sequence_examples([history], _mapping(), max_history=2)
    assert [(row.history_items, row.target_item) for row in examples["train"]] == [
        (("A",), "B"), (("A", "B"), "C"),
    ]
    assert examples["validation"][0].history_items == ("B", "C")
    assert examples["validation"][0].target_item == "D"
    assert examples["test"][0].history_items == ("C", "D")
    assert examples["test"][0].target_item == "E"
    assert all(row.target_item not in {"D", "E"} for row in examples["train"])
    assert examples["train"][1].target_code == _mapping().encode("C")


def test_single_train_item_has_no_training_prefix_but_is_evaluated() -> None:
    history = UserHistory("u", ("A",), "B", "C", (10,), 20, 30)
    examples = build_sequence_examples([history], _mapping())
    assert examples["train"] == []
    assert len(examples["validation"]) == len(examples["test"]) == 1
    assert examples["test"][0].history_items == ("A", "B")


def test_invalid_temporal_order_and_missing_catalog_item_fail() -> None:
    broken = UserHistory("u", ("A", "B"), "C", "D", (20, 10), 30, 40)
    with pytest.raises(ValueError, match="timestamps"):
        build_sequence_examples([broken], _mapping())
    missing = UserHistory("u", ("A",), "Z", "C", (10,), 20, 30)
    with pytest.raises(KeyError, match="Z"):
        build_sequence_examples([missing], _mapping())
