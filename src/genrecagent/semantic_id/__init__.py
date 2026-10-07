"""Public Semantic-ID APIs, loading optional numerical/model runtimes only on demand."""

from importlib import import_module

_EXPORTS = {
    "SemanticCodebook": "codebook",
    "PCAWhitening": "projection",
    "PCAWhiteningConfig": "projection",
    "ResidualQuantizer": "quantizer",
    "ResidualKMeans": "rqkmeans",
    "RQKMeansConfig": "rqkmeans",
    "RQVAE": "rqvae",
    "RQVAEConfig": "rqvae",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
