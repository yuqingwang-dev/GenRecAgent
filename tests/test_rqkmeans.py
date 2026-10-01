import numpy as np
import pytest

from genrecagent.semantic_id.codebook import SemanticCodebook
from genrecagent.semantic_id.projection import PCAWhitening, PCAWhiteningConfig
from genrecagent.semantic_id.rqkmeans import ResidualKMeans, RQKMeansConfig


def test_residual_kmeans_fit_encode_save_load(tmp_path) -> None:
    rng = np.random.default_rng(13)
    centers = np.eye(4, 8, dtype=np.float32)
    vectors = np.vstack([
        center + 0.03 * rng.normal(size=(20, 8)) for center in centers
    ]).astype(np.float32)
    config = RQKMeansConfig(
        input_dim=8, num_levels=2, codebook_size=4,
        batch_size=16, max_iter=20, seed=7,
    )
    model = ResidualKMeans(config).fit(vectors)
    codes, residual = model.encode(vectors)
    assert codes.shape == (80, 2)
    assert residual.shape == vectors.shape
    assert np.isfinite(residual).all()
    assert np.allclose(model.reconstruct(codes) + residual, vectors, atol=1e-6)
    assert len(np.unique(codes[:, 0])) >= 3
    assert float(np.mean(residual**2)) < float(np.mean(vectors**2))
    path = tmp_path / "rqkmeans_codebooks.npz"
    model.save(path)
    loaded = ResidualKMeans.load(path)
    loaded_codes, loaded_residual = loaded.encode(vectors)
    assert np.array_equal(codes, loaded_codes)
    assert np.allclose(residual, loaded_residual)
    with pytest.raises(FileExistsError):
        model.save(path)


def test_kmeans_codes_keep_collision_candidates() -> None:
    mapping = SemanticCodebook(metadata={"model_type": "rq_kmeans_baseline"})
    mapping.add("item-1", ("<s0:0>", "<s1:1>"))
    mapping.add("item-2", ("<s0:0>", "<s1:1>"))
    assert mapping.lookup(("<s0:0>", "<s1:1>")) == ("item-1", "item-2")
    assert mapping.collision_stats()["colliding_item_count"] == 2


def test_projected_codebooks_reload_to_identical_codes(tmp_path) -> None:
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(160, 12)).astype(np.float32)
    train, validation = vectors[:128], vectors[128:]
    projector = PCAWhitening(PCAWhiteningConfig(12, 6, seed=42)).fit(train)
    model = ResidualKMeans(RQKMeansConfig(
        input_dim=6, num_levels=2, codebook_size=8,
        batch_size=32, max_iter=20, seed=42,
    )).fit(projector.transform(train))
    before, _ = model.encode(projector.transform(validation))
    projection_path = tmp_path / "pca_whitening.npz"
    codebook_path = tmp_path / "rqkmeans_codebooks.npz"
    projector.save(projection_path)
    model.save(codebook_path)
    reloaded_projection = PCAWhitening.load(projection_path)
    reloaded_model = ResidualKMeans.load(codebook_path)
    after, _ = reloaded_model.encode(reloaded_projection.transform(validation))
    assert np.array_equal(before, after)
