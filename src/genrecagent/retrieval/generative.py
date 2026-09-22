"""Generative retrieval extension point."""

from typing import Any


class GenerativeRetriever:
    """Future semantic-ID sequence generator."""

    def retrieve(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        del query, top_k
        raise NotImplementedError("Generative retrieval is planned for Phase 3.")

