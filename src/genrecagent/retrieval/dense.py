"""Dense retrieval interface and network-free placeholder."""

from typing import Any


class DenseRetriever:
    """Future embedding-backed retriever."""

    def retrieve(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        del query, top_k
        raise NotImplementedError("Dense retrieval is planned for Phase 1.")

