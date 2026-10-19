"""Retrieval module: stores vectors and finds the nearest feedback for a query vector.

Exposes a ``VectorIndex`` port plus the position -> ``feedback_id`` mapping store.
"""

from rag_poc.retrieval.retrieval_ports import (
    SearchIndex,
    SearchIndexStore,
    VectorIndex,
    VectorIndexFactory,
)

__all__ = ["SearchIndex", "SearchIndexStore", "VectorIndex", "VectorIndexFactory"]
