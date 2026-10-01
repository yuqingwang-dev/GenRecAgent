import torch

from genrecagent.retrieval.semantic_trie import SemanticCodeTrie
from genrecagent.retrieval.transformer import (
    CausalSemanticTransformer,
    SemanticVocabulary,
    TransformerConfig,
    TransformerTokenScorer,
    collate_teacher_forced,
)
from genrecagent.semantic_id.codebook import SemanticCodebook


def _book() -> SemanticCodebook:
    return SemanticCodebook(assignments={
        "A": ("<s0:0>", "<s1:0>", "<s2:0>"),
        "B": ("<s0:1>", "<s1:1>", "<s2:1>"),
    })


def test_teacher_forcing_masks_history_and_padding() -> None:
    book = _book()
    vocab = SemanticVocabulary.from_codebook(book)
    examples = [
        {"history_codes": [book.encode("A")], "target_code": book.encode("B")},
        {"history_codes": [book.encode("A"), book.encode("B")],
         "target_code": book.encode("A")},
    ]
    inputs, labels = collate_teacher_forced(examples, vocab, max_history=20)
    assert inputs.shape == labels.shape
    assert labels.ne(-100).sum(dim=1).tolist() == [3, 3]
    assert labels[0, :4].eq(-100).all()
    assert inputs[0, 4] == vocab.predict_id
    assert labels[0, 4] == vocab.to_id["<s0:1>"]
    assert labels[0, -1] == -100
    assert inputs[0, -1] == vocab.pad_id
    assert inputs[0, 6] == vocab.to_id["<s1:1>"]
    assert vocab.to_id["<s2:1>"] not in inputs[0].tolist()


def test_causal_model_gradients_save_load_and_valid_decode(tmp_path) -> None:
    torch.manual_seed(5)
    book = _book()
    vocab = SemanticVocabulary.from_codebook(book)
    model = CausalSemanticTransformer(TransformerConfig(
        vocab_size=vocab.size, max_positions=15, pad_id=vocab.pad_id,
        d_model=16, n_heads=4, n_layers=1, feedforward_dim=32, dropout=0.0,
    ))
    examples = [{"history_codes": [book.encode("A")], "target_code": book.encode("B")}]
    inputs, labels = collate_teacher_forced(examples, vocab, max_history=3)
    loss = model.target_loss(inputs, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.token_embedding.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        original = model(inputs)
        changed = inputs.clone()
        changed[0, -1] = vocab.to_id["<s1:0>"]
        comparison = model(changed)
    assert torch.allclose(original[:, :-1], comparison[:, :-1], atol=1e-6)
    path = tmp_path / "transformer.pt"
    model.save(path, vocab, max_history=3)
    restored, restored_vocab, max_history = CausalSemanticTransformer.load(path)
    restored.eval()
    assert restored_vocab.tokens == vocab.tokens
    assert max_history == 3
    with torch.no_grad():
        assert torch.allclose(original, restored(inputs), atol=1e-6)
    scorer = TransformerTokenScorer(restored, restored_vocab, max_history=3, device="cpu")
    batch_scores = scorer.log_probabilities_batch(
        (book.encode("A"),), ((), ("<s0:0>",))
    )
    assert len(batch_scores) == 2
    assert batch_scores[0] == scorer.log_probabilities((book.encode("A"),), ())
    predictions = SemanticCodeTrie(book.assignments.values()).beam_search(
        scorer, (book.encode("A"),), beam_size=2, max_codes=2
    )
    assert len(predictions) == 2
    assert all(book.lookup(row.semantic_id) for row in predictions)
