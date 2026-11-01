"""Retrieval module: stores vectors and finds the nearest feedback for a query.

Exposes a ``VectorIndex`` port, the position -> ``feedback_id`` mapping store, and
``SemanticRetriever``, the reusable search service built on them.
"""

from rag_poc.retrieval.retrieval_ports import (
    FeedbackRetriever,
    QueryEmbedder,
    SearchIndex,
    SearchIndexStore,
    VectorIndex,
    VectorIndexFactory,
)
from rag_poc.retrieval.semantic_retriever import SemanticRetriever

__all__ = [
    "FeedbackRetriever",
    "QueryEmbedder",
    "SearchIndex",
    "SearchIndexStore",
    "SemanticRetriever",
    "VectorIndex",
    "VectorIndexFactory",
]
