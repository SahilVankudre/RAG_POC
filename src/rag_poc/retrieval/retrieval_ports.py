"""Ports of the retrieval module."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from rag_poc.lib import FeedbackItem, SearchFilters, SearchIndexManifest, SearchResult


class VectorIndex(Protocol):
    """Nearest-neighbour search over vectors addressed by their insertion position."""

    @property
    def dimension(self) -> int: ...

    @property
    def size(self) -> int: ...

    @property
    def index_type(self) -> str: ...

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Return up to ``top_k`` (position, score) pairs, best first."""
        ...

    def write(self, location: Path) -> None: ...


VectorIndexFactory = Callable[[np.ndarray], VectorIndex]
"""Builds a ``VectorIndex`` from an ``(n, dimension)`` float32 matrix; row i -> position i."""


class QueryEmbedder(Protocol):
    """The part of an embedding model that search needs.

    Declared here so retrieval stays independent of the embedding module: any embedder with
    these members fits, without either module importing the other.
    """

    @property
    def model_name(self) -> str: ...

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one search query, shape ``(dimension,)``."""
        ...


@dataclass(frozen=True, slots=True)
class SearchIndex:
    """A vector index plus the feedback each position refers to (``feedback[i]`` <-> position i)."""

    vectors: VectorIndex
    feedback: Sequence[FeedbackItem]
    manifest: SearchIndexManifest


class FeedbackRetriever(Protocol):
    """Finds the feedback most relevant to a natural-language query."""

    def search(
        self, query: str, top_k: int | None = None, filters: SearchFilters | None = None
    ) -> Sequence[SearchResult]: ...


class SearchIndexStore(Protocol):
    """Persists the vector index, its position -> feedback mapping and its manifest together."""

    @property
    def location(self) -> Path: ...

    def exists(self) -> bool: ...

    def read_manifest(self) -> SearchIndexManifest | None: ...

    def save(self, search_index: SearchIndex) -> None: ...

    def load(self) -> SearchIndex: ...
