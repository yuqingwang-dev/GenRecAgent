from genrecagent.data.models import UserHistory
from genrecagent.retrieval.generative import (
    GenerativeRetriever,
    expand_code_predictions,
    train_popularity_counts,
)
from genrecagent.retrieval.semantic_trie import CodePrediction, SemanticCodeTrie
from genrecagent.semantic_id.codebook import SemanticCodebook

COLLISION = ("<s0:0>", "<s1:0>", "<s2:0>")
SECOND = ("<s0:0>", "<s1:1>", "<s2:0>")
THIRD = ("<s0:1>", "<s1:0>", "<s2:0>")


class FixedScorer:
    def log_probabilities(self, history, prefix):
        del history
        choices = {
            (): {"<s0:0>": -0.1, "<s0:1>": -1.0, "<s0:99>": 100.0},
            ("<s0:0>",): {"<s1:0>": -0.1, "<s1:1>": -0.5},
            ("<s0:1>",): {"<s1:0>": -0.1},
            ("<s0:0>", "<s1:0>"): {"<s2:0>": -0.1},
            ("<s0:0>", "<s1:1>"): {"<s2:0>": -0.1},
            ("<s0:1>", "<s1:0>"): {"<s2:0>": -0.1},
        }
        return choices.get(prefix, {})


class NoValidScorer:
    def log_probabilities(self, history, prefix):
        del history, prefix
        return {"invalid": 0.0}


def _mapping() -> SemanticCodebook:
    return SemanticCodebook(assignments={
        "A": COLLISION, "B": COLLISION, "C": SECOND, "D": THIRD,
    })


def test_trie_masks_invalid_tokens_and_returns_catalog_codes() -> None:
    trie = SemanticCodeTrie(_mapping().assignments.values())
    assert trie.allowed_next(()) == ("<s0:0>", "<s0:1>")
    assert trie.allowed_next(("<s0:99>",)) == ()
    predictions = trie.beam_search(FixedScorer(), (), beam_size=3, max_codes=3)
    assert [row.semantic_id for row in predictions] == [COLLISION, SECOND, THIRD]
    assert all(trie.is_valid(row.semantic_id) for row in predictions)
    assert trie.beam_search(NoValidScorer(), (), beam_size=2, max_codes=2) == ()


def test_collision_expansion_uses_train_counts_and_filters_seen() -> None:
    histories = [
        UserHistory("u1", ("B", "A"), "C", "D", (1, 2), 3, 4),
        UserHistory("u2", ("B",), "C", "D", (1,), 2, 3),
    ]
    counts = train_popularity_counts(histories)
    assert counts["B"] == 2
    assert counts["C"] == 0
    result = GenerativeRetriever(
        _mapping(), FixedScorer(), train_popularity=counts, beam_size=3, max_codes=3
    ).retrieve(("B",), top_k=3)
    assert [hit.item_id for hit in result.hits] == ["A", "C", "D"]
    assert all(hit.source == "generative" for hit in result.hits)
    assert [hit.rank for hit in result.hits] == [1, 2, 3]
    assert result.hit_codes[0] == COLLISION
    assert result.diagnostics.collision_codes == 1
    assert result.diagnostics.excluded_seen == 1


def test_expansion_records_invalid_duplicate_and_shortfall() -> None:
    predictions = (
        CodePrediction(COLLISION, -0.1, 1),
        CodePrediction(COLLISION, -0.2, 2),
        CodePrediction(("invalid",), -0.3, 3),
    )
    result = expand_code_predictions(
        predictions, _mapping(), train_popularity={}, seen_items={"A"}, top_k=3
    )
    assert [hit.item_id for hit in result.hits] == ["B"]
    assert result.diagnostics.duplicate_codes == 1
    assert result.diagnostics.invalid_codes == 1
    assert result.diagnostics.candidate_shortfall == 2
    empty = GenerativeRetriever(
        _mapping(), NoValidScorer(), train_popularity={}
    ).retrieve((), top_k=2)
    assert empty.hits == ()
    assert empty.diagnostics.candidate_shortfall == 2
