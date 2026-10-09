from dataclasses import asdict

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from genrecagent.semantic_id.projection import PCAWhitening
from genrecagent.semantic_id.quality import checkpoint_score, code_statistics, quality_gate
from genrecagent.semantic_id.quantizer import ResidualQuantizer
from genrecagent.semantic_id.rqvae import RQVAE, RQVAEConfig
from genrecagent.semantic_id.stabilized import variance_covariance
from genrecagent.semantic_id.training import TrainingConfig, fit_rqvae, split_item_indices


def _config():
    return RQVAEConfig(
        input_dim=8,
        hidden_dim=16,
        latent_dim=4,
        num_levels=2,
        codebook_size=8,
        variant="pca_ema",
        quantization_weight=0.05,
        variance_weight=0.1,
        covariance_weight=0.01,
    )


def _vectors(n=96):
    values = np.random.default_rng(4).normal(size=(n, 8)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_stabilized_encoder_decoder_and_ema_really_learn_and_save(tmp_path):
    vectors = _vectors()
    model = RQVAE(_config())
    model.initialize_pca(vectors[:64], seed=42)
    model.quantizer.initialize_from_latents(
        model.encoder(torch.from_numpy(vectors[:64])), seed=42, lloyd_iterations=2
    )
    encoder_before = model.encoder.projection.weight.detach().clone()
    decoder_before = model.decoder.projection.weight.detach().clone()
    book_before = model.quantizer.codebooks[0].weight.detach().clone()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.001)
    model.train()
    output = model(torch.from_numpy(vectors[64:]))
    output.total_loss.backward()
    optimizer.step()
    assert not torch.equal(encoder_before, model.encoder.projection.weight)
    assert not torch.equal(decoder_before, model.decoder.projection.weight)
    assert not torch.equal(book_before, model.quantizer.codebooks[0].weight)
    assert not model.quantizer.codebooks[0].weight.requires_grad
    assert torch.isfinite(output.total_loss) and model.quantizer.ema_updates == 1
    model.eval()
    path = tmp_path / "stable.pt"
    model.save(path)
    restored = RQVAE.load(path).eval()
    data = torch.from_numpy(vectors[:16])
    assert torch.equal(model.encode(data), restored.encode(data))
    assert torch.allclose(model.decode(model.encode(data)), restored.decode(restored.encode(data)))


def test_validation_and_encode_cannot_update_ema_even_when_encoder_is_training():
    vectors = _vectors()
    model = RQVAE(_config())
    model.initialize_pca(vectors, seed=42)
    model.quantizer.initialize_from_latents(model.encoder(torch.from_numpy(vectors)), seed=42)
    before = {key: value.clone() for key, value in model.quantizer.state_dict().items()}
    model.eval()
    model(torch.from_numpy(vectors))
    model.train()
    model.encode(torch.from_numpy(vectors))
    assert all(
        torch.equal(value, model.quantizer.state_dict()[key]) for key, value in before.items()
    )


def test_dead_codes_restart_only_from_current_training_residuals():
    quantizer = ResidualQuantizer(
        4, 1, 8, update_method="ema", ema_decay=0.9, dead_code_threshold=0.2
    )
    with torch.no_grad():
        quantizer.codebooks[0].weight.zero_()
        quantizer.ema_counts.zero_()
        quantizer.ema_sums.zero_()
    residuals = torch.ones(16, 4)
    quantizer.train()
    quantizer(residuals)
    assert int(quantizer.dead_resets[0]) == 7
    assert torch.allclose(quantizer.codebooks[0].weight, torch.ones(8, 4))


def test_old_checkpoint_schema_remains_loadable_and_equivalent(tmp_path):
    model = RQVAE(
        RQVAEConfig(input_dim=8, hidden_dim=16, latent_dim=4, num_levels=2, codebook_size=8)
    ).eval()
    original_fields = {
        key: asdict(model.config)[key]
        for key in (
            "input_dim",
            "hidden_dim",
            "latent_dim",
            "num_levels",
            "codebook_size",
            "commitment_weight",
        )
    }
    path = tmp_path / "legacy.pt"
    torch.save({"config": original_fields, "state_dict": model.state_dict()}, path)
    loaded = RQVAE.load(path).eval()
    assert loaded.config.variant == "legacy"
    values = torch.from_numpy(_vectors())
    assert torch.equal(model.encode(values), loaded.encode(values))


def test_pca_and_initialization_only_see_training_items(monkeypatch):
    vectors, ids = _vectors(), [f"item-{i}" for i in range(96)]
    train, validation = split_item_indices(ids, seed=42, validation_fraction=0.25)
    permitted = {row.tobytes() for row in vectors[train]}
    forbidden = {row.tobytes() for row in vectors[validation]}
    original_fit = PCAWhitening.fit

    def checked_fit(self, values):
        rows = {row.tobytes() for row in values}
        assert rows <= permitted and not rows & forbidden
        return original_fit(self, values)

    monkeypatch.setattr(PCAWhitening, "fit", checked_fit)
    result = fit_rqvae(
        vectors,
        ids,
        _config(),
        TrainingConfig(
            device="cpu",
            epochs=2,
            warmup_epochs=1,
            batch_size=16,
            validation_fraction=0.25,
            pca_fit_samples=48,
            init_samples=48,
            lloyd_iterations=2,
            gradient_clip=1,
            min_utilization=0.25,
            min_unique_ratio=0.5,
        ),
    )
    assert result.initialization["pca_item_count"] == 48
    assert len(result.warmup_log) == 1 and len(result.epoch_log) == 2
    assert result.selected_validation_quality["gate"]["enabled"]


def test_collapse_and_redundant_dimensions_receive_regularization():
    torch.manual_seed(3)
    random = torch.randn(128, 4)
    constant = torch.ones(128, 4)
    correlated = torch.randn(128, 1).repeat(1, 4)
    assert variance_covariance(constant)[0] > variance_covariance(random)[0]
    assert variance_covariance(correlated)[1] > variance_covariance(random)[1]


def test_quality_selection_rejects_low_mse_collapsed_codes():
    collapsed = code_statistics(np.zeros((64, 2), dtype=np.int64), 8)
    healthy = code_statistics(np.array([(i, j) for i in range(8) for j in range(8)]), 8)
    limits = dict(min_utilization=0.5, min_perplexity_fraction=0.25, min_unique_ratio=0.5)
    bad, good = quality_gate(collapsed, **limits), quality_gate(healthy, **limits)
    assert not bad["passed"] and good["passed"]
    assert checkpoint_score(healthy, good, 0.003) > checkpoint_score(collapsed, bad, 0.00001)
    with pytest.raises(ValueError):
        checkpoint_score(healthy, good, float("nan"))


def test_stabilized_forward_requires_explicit_training_initialization():
    model = RQVAE(_config())
    with pytest.raises(RuntimeError, match="Initialize"):
        model(F.normalize(torch.randn(8, 8), dim=1))
