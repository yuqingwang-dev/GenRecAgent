import pytest

from genrecagent.retrieval.rank_fusion import fuse_code_ranks
from genrecagent.retrieval.semantic_trie import CodePrediction

A = ("a",)
B = ("b",)
C = ("c",)


def _prediction(code: tuple[str, ...], rank: int) -> CodePrediction:
    return CodePrediction(code, -float(rank), rank)


def test_fusion_endpoints_preserve_original_rankings() -> None:
    transition = (_prediction(A, 1), _prediction(B, 2))
    transformer = (_prediction(C, 1), _prediction(B, 2))
    assert fuse_code_ranks(
        transition, transformer, transformer_weight=0, max_codes=2
    ) == transition
    assert fuse_code_ranks(
        transition, transformer, transformer_weight=1, max_codes=2
    ) == transformer


def test_fusion_rewards_agreement_and_is_deterministic() -> None:
    transition = (_prediction(A, 1), _prediction(B, 2))
    transformer = (_prediction(C, 1), _prediction(B, 2))
    fused = fuse_code_ranks(
        transition, transformer, transformer_weight=0.5, max_codes=3
    )
    assert [row.semantic_id for row in fused] == [B, A, C]
    assert [row.rank for row in fused] == [1, 2, 3]
    assert all(row.score > 0 for row in fused)


def test_fusion_rejects_invalid_configuration_and_duplicate_codes() -> None:
    with pytest.raises(ValueError):
        fuse_code_ranks((), (), transformer_weight=-0.1, max_codes=1)
    with pytest.raises(ValueError):
        fuse_code_ranks((), (), transformer_weight=0.5, max_codes=0)
    with pytest.raises(ValueError, match="unique"):
        fuse_code_ranks(
            (_prediction(A, 1), _prediction(A, 2)), (),
            transformer_weight=0.5, max_codes=2,
        )
