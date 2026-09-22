"""Retrieval extension points."""

from genrecagent.retrieval.dense import DenseRetriever
from genrecagent.retrieval.generative import GenerativeRetriever
from genrecagent.retrieval.reranker import Reranker

__all__ = ["DenseRetriever", "GenerativeRetriever", "Reranker"]

