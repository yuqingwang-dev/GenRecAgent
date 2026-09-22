"""Reranking extension point."""

from typing import Any


class Reranker:
    """Future learned or LLM-based reranker."""

    def rerank(self, query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        del query, items
        raise NotImplementedError("Reranking is planned for a later phase.")

