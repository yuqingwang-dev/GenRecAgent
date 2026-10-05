"""Exact cosine retrieval over normalized item embeddings."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from genrecagent.retrieval.encoder import normalize_rows, normalize_vector


@dataclass(frozen=True, slots=True)
class SearchHit:
    item_id: str
    score: float
    rank: int
    source: str

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


class DenseRetriever:
    """Flat inner-product index; NumPy exact search or optional FAISS IndexFlatIP."""

    def __init__(self, backend: str = "numpy") -> None:
        if backend not in {"numpy", "faiss"}:
            raise ValueError("backend must be numpy or faiss")
        self.backend = backend
        self.item_ids: list[str] = []
        self.item_to_index: dict[str, int] = {}
        self._id_array = np.empty(0, dtype=str)
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self._index: Any = None
        self.metadata: dict[str, Any] = {}

    def build_index(
        self,
        item_ids: list[str],
        embeddings: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or not len(item_ids) or matrix.shape[0] != len(item_ids):
            raise ValueError("Item IDs and a nonempty 2D embedding matrix must align")
        if len(set(item_ids)) != len(item_ids) or not np.isfinite(matrix).all():
            raise ValueError("Item IDs must be unique and embeddings finite")
        self.item_ids = list(item_ids)
        self.item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
        self._id_array = np.asarray(item_ids)
        self.embeddings = normalize_rows(matrix)
        if np.any(np.linalg.norm(self.embeddings, axis=1) == 0):
            raise ValueError("Zero item embeddings cannot support cosine retrieval")
        self.metadata = dict(metadata or {})
        self.metadata.update(
            {
                "embedding_dimension": int(matrix.shape[1]),
                "item_count": len(item_ids),
                "normalization": "l2",
                "similarity": "inner_product",
                "index_type": "flat_ip",
                "backend": self.backend,
            }
        )
        if self.backend == "faiss":
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError("faiss-cpu unavailable; use backend=numpy") from exc
            self._index = faiss.IndexFlatIP(matrix.shape[1])
            self._index.add(self.embeddings)

    def save_index(self, path: Path) -> None:
        if not self.item_ids:
            raise RuntimeError("Build an index before saving it")
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Index directory is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "item_embeddings.npy", self.embeddings)
        (path / "item_ids.json").write_text(json.dumps(self.item_ids), encoding="utf-8")
        (path / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2), encoding="utf-8"
        )
        if self.backend == "faiss":
            import faiss

            faiss.write_index(self._index, str(path / "index.faiss"))

    @classmethod
    def load_index(cls, path: Path, *, backend: str | None = None) -> "DenseRetriever":
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        saved_backend = metadata["backend"]
        retriever = cls(backend=backend or saved_backend)
        retriever.build_index(
            json.loads((path / "item_ids.json").read_text(encoding="utf-8")),
            np.load(path / "item_embeddings.npy", allow_pickle=False),
            metadata=metadata,
        )
        if retriever.backend == "faiss" and saved_backend == "faiss":
            import faiss

            retriever._index = faiss.read_index(str(path / "index.faiss"))
            if retriever._index.ntotal != len(retriever.item_ids):
                raise ValueError("FAISS index and item mapping have different lengths")
        return retriever

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 10,
        exclude_seen_items: set[str] | frozenset[str] | None = None,
        allowed_item_ids: set[str] | frozenset[str] | None = None,
    ) -> list[SearchHit]:
        if not self.item_ids or top_k <= 0:
            return []
        vector = normalize_vector(query_embedding)
        if vector is None or len(vector) != self.embeddings.shape[1]:
            raise ValueError("Query must be nonzero and match embedding dimension")
        seen = exclude_seen_items or set()
        if self.backend == "faiss" and allowed_item_ids is None:
            requested = min(len(self.item_ids), top_k + len(seen))
            scores, indices = self._index.search(vector.reshape(1, -1), requested)
            ordered = [
                (int(index), float(score))
                for index, score in zip(indices[0], scores[0], strict=True)
            ]
        else:
            scores = self.embeddings @ vector
            for item_id in seen:
                index = self.item_to_index.get(item_id)
                if index is not None:
                    scores[index] = -np.inf
            eligible_count = len(self.item_ids) - len(seen & self.item_to_index.keys())
            if allowed_item_ids is not None:
                eligible = np.zeros(len(self.item_ids), dtype=bool)
                for item_id in allowed_item_ids:
                    index = self.item_to_index.get(item_id)
                    if index is not None and item_id not in seen:
                        eligible[index] = True
                scores[~eligible] = -np.inf
                eligible_count = int(eligible.sum())
            count = min(top_k, eligible_count)
            if count == 0:
                return []
            if count < len(scores):
                partition = np.argpartition(scores, -count)[-count:]
                threshold = scores[partition].min()
                candidates = np.flatnonzero(scores >= threshold)
            else:
                candidates = np.flatnonzero(np.isfinite(scores))
            ranked = candidates[np.lexsort((self._id_array[candidates], -scores[candidates]))]
            ordered = [
                (int(index), float(scores[index]))
                for index in ranked[:count]
            ]
        hits: list[SearchHit] = []
        for index, score in ordered:
            if index >= 0 and self.item_ids[index] not in seen:
                hits.append(SearchHit(self.item_ids[index], score, len(hits) + 1, "dense"))
                if len(hits) >= top_k:
                    break
        return hits


def mean_user_embedding(
    history: tuple[str, ...] | list[str], retriever: DenseRetriever
) -> np.ndarray | None:
    """Pool only observed history items; unknown IDs do not inject zero vectors."""

    indices = [
        retriever.item_to_index[item_id]
        for item_id in history
        if item_id in retriever.item_to_index
    ]
    if not indices:
        return None
    return normalize_vector(retriever.embeddings[indices].mean(axis=0))
