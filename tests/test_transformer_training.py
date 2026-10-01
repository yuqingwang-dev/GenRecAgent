import math

import pytest

from genrecagent.retrieval.transformer import SemanticVocabulary, TransformerConfig
from genrecagent.retrieval.transformer_training import (
    TransformerTrainingConfig,
    fit_semantic_transformer,
)
from genrecagent.semantic_id.codebook import SemanticCodebook


def _examples(split: str, count: int) -> list[dict]:
    return [
        {
            "split": split,
            "history_codes": [["<s0:0>", "<s1:0>", "<s2:0>"]],
            "target_code": ["<s0:1>", "<s1:1>", "<s2:1>"],
        }
        for _ in range(count)
    ]


def test_one_epoch_cpu_training_selects_validation_checkpoint() -> None:
    book = SemanticCodebook(assignments={
        "A": ("<s0:0>", "<s1:0>", "<s2:0>"),
        "B": ("<s0:1>", "<s1:1>", "<s2:1>"),
    })
    vocab = SemanticVocabulary.from_codebook(book)
    config = TransformerConfig(
        vocab_size=vocab.size, max_positions=11, pad_id=vocab.pad_id,
        d_model=16, n_heads=4, n_layers=1, feedforward_dim=32, dropout=0.0,
    )
    result = fit_semantic_transformer(
        _examples("train", 8), _examples("validation", 4), vocab,
        config,
        TransformerTrainingConfig(epochs=1, batch_size=4, device="cpu"),
        max_history=2,
    )
    assert result.best_epoch == 1
    assert result.device == "cpu"
    assert math.isfinite(result.best_validation_loss)
    assert len(result.epoch_log) == 1
    assert 0 <= result.epoch_log[0]["validation_token_accuracy"] <= 1


def test_training_refuses_validation_as_training_data() -> None:
    book = SemanticCodebook(assignments={"A": ("0", "1", "2")})
    vocab = SemanticVocabulary.from_codebook(book)
    config = TransformerConfig(vocab_size=vocab.size, max_positions=7,
                               pad_id=vocab.pad_id, d_model=8, n_heads=2,
                               n_layers=1, feedforward_dim=16)
    with pytest.raises(ValueError, match="train-prefix"):
        fit_semantic_transformer(
            _examples("validation", 2), _examples("validation", 1),
            vocab, config, TransformerTrainingConfig(epochs=1, device="cpu"),
            max_history=1,
        )
