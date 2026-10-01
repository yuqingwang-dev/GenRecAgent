import math

import numpy as np
import torch
from torch.nn import functional as F

from genrecagent.semantic_id.codebook import SemanticCodebook
from genrecagent.semantic_id.rqvae import RQVAE, RQVAEConfig
from genrecagent.semantic_id.training import (
    TrainingConfig,
    encode_catalog,
    fit_rqvae,
    split_item_indices,
)


def test_rqvae_has_learned_codebook_and_gradients() -> None:
    torch.manual_seed(3)
    model = RQVAE(RQVAEConfig(input_dim=8, hidden_dim=16, latent_dim=4,
                              num_levels=2, codebook_size=8))
    vectors = F.normalize(torch.randn(12, 8), dim=1)
    output = model(vectors)
    assert output.reconstruction.shape == (12, 8)
    assert output.codes.shape == (12, 2)
    assert int(output.codes.min()) >= 0
    assert int(output.codes.max()) < 8
    output.total_loss.backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.decoder[0].weight.grad.abs().sum() > 0
    assert any(book.weight.grad.abs().sum() > 0 for book in model.quantizer.codebooks)


def test_training_split_depends_on_ids_and_seed() -> None:
    ids = [f"item-{index}" for index in range(20)]
    train, validation = split_item_indices(ids, seed=17, validation_fraction=0.2)
    assert len(train) == 16
    assert len(validation) == 4
    assert set(train).isdisjoint(set(validation))
    reversed_ids = list(reversed(ids))
    _, reversed_validation = split_item_indices(
        reversed_ids, seed=17, validation_fraction=0.2
    )
    assert {ids[index] for index in validation} == {
        reversed_ids[index] for index in reversed_validation
    }


def test_cpu_smoke_train_encode_save_and_load(tmp_path) -> None:
    rng = np.random.default_rng(9)
    vectors = rng.normal(size=(64, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"item-{index:03d}" for index in range(64)]
    config = RQVAEConfig(input_dim=8, hidden_dim=16, latent_dim=4,
                         num_levels=2, codebook_size=8)
    result = fit_rqvae(
        vectors, ids, config,
        TrainingConfig(epochs=2, batch_size=16, validation_fraction=0.25, device="cpu"),
    )
    assert 1 <= result.best_epoch <= 2
    assert math.isfinite(result.best_validation_reconstruction_mse)
    assert len(result.train_indices) == 48
    assert len(result.validation_indices) == 16
    mapping, metrics = encode_catalog(result.model, vectors, ids, batch_size=16, device="cpu")
    assert len(mapping.assignments) == 64
    assert len(metrics["codebook_usage_by_level"]) == 2
    assert math.isfinite(metrics["catalog_reconstruction_mse"])
    assert metrics["collisions"]["item_count"] == 64
    assert all(mapping.lookup(mapping.encode(item_id)) for item_id in ids)
    model_path = tmp_path / "rqvae.pt"
    map_path = tmp_path / "item_semantic_ids.json"
    result.model.save(model_path)
    mapping.save(map_path)
    restored = RQVAE.load(model_path)
    restored.eval()
    with torch.no_grad():
        assert torch.equal(
            result.model.encode(torch.from_numpy(vectors[:8])),
            restored.encode(torch.from_numpy(vectors[:8])),
        )
        decoded = restored.decode(restored.encode(torch.from_numpy(vectors[:8])))
        assert decoded.shape == (8, 8)
        assert torch.isfinite(decoded).all()
    loaded_mapping = SemanticCodebook.load(map_path)
    assert loaded_mapping.assignments == mapping.assignments
    assert loaded_mapping.lookup(mapping.encode(ids[0])) == mapping.lookup(mapping.encode(ids[0]))


def test_collisions_keep_all_items_and_reassignment_updates_reverse_map() -> None:
    mapping = SemanticCodebook()
    code = ("<s0:1>", "<s1:3>")
    mapping.add("B", code)
    mapping.add("A", code)
    assert mapping.lookup(code) == ("A", "B")
    assert mapping.collision_stats()["colliding_item_count"] == 2
    mapping.add("A", ("<s0:2>", "<s1:3>"))
    assert mapping.lookup(code) == ("B",)
    assert mapping.lookup(("<s0:99>",)) == ()
