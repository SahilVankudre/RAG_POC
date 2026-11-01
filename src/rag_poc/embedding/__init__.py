"""Embedding module: converts text into normalized semantic vectors.

Exposes an ``Embedder`` port; concrete model adapters are built by the composition root.
"""

from rag_poc.embedding.batch_embedding import (
    ProgressCallback,
    embed_in_batches,
    fingerprint_embedding_input,
)
from rag_poc.embedding.embedding_ports import Embedder, EmbeddingStore

__all__ = [
    "Embedder",
    "EmbeddingStore",
    "ProgressCallback",
    "embed_in_batches",
    "fingerprint_embedding_input",
]
