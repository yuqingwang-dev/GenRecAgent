import math

import pytest
from scripts.evaluate_transition import _evaluate

from genrecagent.retrieval.generative import GenerativeRetriever
from genrecagent.retrieval.transition import CodeTransitionScorer
from genrecagent.semantic_id.codebook import SemanticCodebook


def _catalog() -> SemanticCodebook:
    return SemanticCodebook(assignments={
        "A": ("<s0:0>", "<s1:0>", "<s2:0>"),
        "B": ("<s0:1>", "<s1:0>", "<s2:0>"),
        "C": ("<s0:1>", "<s1:1>", "<s2:0>"),
        "D": ("<s0:0>", "<s1:1>", "<s2:1>"),
    })


def _example(source: str, target: str, *, split: str = "train") -> dict[str, object]:
    book = _catalog()
    return {
        "split": split,
        "history_codes": [book.encode(source)],
        "target_code": book.encode(target),
    }


def test_transition_uses_train_only_and_returns_normalized_level_probabilities(tmp_path) -> None:
    book = _catalog()
    scorer = CodeTransitionScorer(book, alpha=2.0)
    with pytest.raises(RuntimeError):
        scorer.log_probabilities((book.encode("A"),), ())
    with pytest.raises(ValueError, match="train"):
        scorer.fit([_example("A", "B", split="validation")])
    impossible = _example("A", "B")
    impossible["target_code"] = ("<s0:0>", "<s1:0>", "<s2:1>")
    with pytest.raises(ValueError, match="catalog"):
        scorer.fit([impossible])
    scorer.fit([_example("A", "B"), _example("A", "B"), _example("B", "D")])
    first = scorer.log_probabilities((book.encode("A"),), ())
    assert first["<s0:1>"] > first["<s0:0>"]
    assert sum(math.exp(value) for value in first.values()) == pytest.approx(1.0)
    second = scorer.log_probabilities((book.encode("A"),), ("<s0:1>",))
    assert sum(math.exp(value) for value in second.values()) == pytest.approx(1.0)
    path = tmp_path / "transition.json"
    scorer.save(path)
    restored = CodeTransitionScorer.load(path, book)
    assert restored.train_examples == 3
    assert restored.log_probabilities((book.encode("A"),), ()) == first


def test_transition_integrates_with_catalog_valid_collision_adapter() -> None:
    book = _catalog()
    scorer = CodeTransitionScorer(book).fit([
        _example("A", "B"), _example("A", "C"), _example("B", "D")
    ])
    retriever = GenerativeRetriever(
        book, scorer, train_popularity={"B": 2, "C": 1}, beam_size=4, max_codes=4
    )
    result = retriever.retrieve(("A",), top_k=3)
    assert len(result.hits) == 3
    assert all(book.lookup(code.semantic_id) for code in result.predicted_codes)
    assert all(hit.item_id != "A" for hit in result.hits)
    assert result.diagnostics.invalid_codes == 0
    metrics = _evaluate(
        [{"target_item": "B", "target_code": list(book.encode("B")),
          "history_items": ["A"]}],
        retriever,
        [1, 3],
        {book.encode("B")},
        "validation",
    )
    assert metrics["users"] == 1
    assert metrics["target_code_seen_in_train_fraction"] == 1.0
    assert metrics["diagnostic_totals"]["invalid_codes"] == 0
