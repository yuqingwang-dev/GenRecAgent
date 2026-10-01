import numpy as np
import pytest
from sklearn.decomposition import PCA

from genrecagent.semantic_id.projection import PCAWhitening, PCAWhiteningConfig


def test_train_only_pca_whitening_save_load(tmp_path) -> None:
    rng = np.random.default_rng(4)
    train = rng.normal(size=(256, 12)).astype(np.float32)
    validation = rng.normal(size=(32, 12)).astype(np.float32)
    config = PCAWhiteningConfig(input_dim=12, output_dim=6, seed=17)
    projector = PCAWhitening(config).fit(train)
    expected = PCA(
        n_components=6, whiten=True, svd_solver="randomized", random_state=17
    ).fit(train).transform(validation)
    transformed = projector.transform(validation)
    assert transformed.shape == (32, 6)
    assert np.allclose(transformed, expected, atol=1e-5)
    reconstructed = projector.inverse_transform(transformed)
    assert reconstructed.shape == validation.shape
    assert np.isfinite(reconstructed).all()
    path = tmp_path / "pca_whitening.npz"
    projector.save(path)
    loaded = PCAWhitening.load(path)
    assert np.allclose(transformed, loaded.transform(validation))
    with pytest.raises(FileExistsError):
        projector.save(path)


def test_projection_rejects_wrong_input_shape() -> None:
    model = PCAWhitening(PCAWhiteningConfig(input_dim=4, output_dim=2))
    with pytest.raises(RuntimeError):
        model.transform(np.zeros((2, 4), dtype=np.float32))
    model.fit(np.eye(4, dtype=np.float32))
    with pytest.raises(ValueError):
        model.transform(np.zeros((2, 3), dtype=np.float32))
