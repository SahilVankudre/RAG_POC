"""Ports of the embedding module."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from rag_poc.lib import FeedbackEmbeddings


class Embedder(Protocol):
    """Turns texts into L2-normalized float32 vectors of a fixed dimension."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents: shape ``(len(texts), dimension)``; row ``i`` encodes ``texts[i]``."""
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one search query, shape ``(dimension,)``.

        Separate from ``embed`` because retrieval models such as BGE expect queries to carry
        an instruction prefix that documents must not have.
        """
        ...


class EmbeddingStore(Protocol):
    """Persists embeddings so they are computed once and reused by indexing and search."""

    @property
    def location(self) -> Path: ...

    def exists(self) -> bool: ...

    def stored_fingerprint(self) -> str | None:
        """Fingerprint of the input the stored embeddings were computed from, if any."""
        ...

    def save(self, embeddings: FeedbackEmbeddings, *, fingerprint: str) -> None: ...

    def load(self) -> FeedbackEmbeddings: ...
