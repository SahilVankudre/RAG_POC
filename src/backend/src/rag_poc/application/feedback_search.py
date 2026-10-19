"""Use case: semantic search over the indexed feedback (spec §8). No LLM involved."""

import logging

import numpy as np

from rag_poc.embedding import Embedder
from rag_poc.lib import (
    FeedbackItem,
    RetrievalError,
    SearchFilters,
    SearchIndexManifest,
    SearchResult,
)
from rag_poc.retrieval import SearchIndex, SearchIndexStore

logger = logging.getLogger(__name__)


class FeedbackSearch:
    """Embeds a natural-language query and returns the most similar feedback.

    The index is loaded from disk on first use and kept in memory, so repeated queries
    (e.g. an interactive session) only pay for embedding the query itself.
    """

    def __init__(
        self,
        *,
        index_store: SearchIndexStore,
        embedder: Embedder,
        default_top_k: int,
        default_filters: SearchFilters | None = None,
    ) -> None:
        self._index_store = index_store
        self._embedder = embedder
        self._default_top_k = default_top_k
        self._default_filters = default_filters or SearchFilters()
        self._loaded: SearchIndex | None = None

    @property
    def default_filters(self) -> SearchFilters:
        return self._default_filters

    def search(
        self, query: str, top_k: int | None = None, filters: SearchFilters | None = None
    ) -> list[SearchResult]:
        """Return up to ``top_k`` results that pass ``filters``, ranked by cosine similarity."""
        query = query.strip()
        if not query:
            raise RetrievalError("Search query must not be empty.")
        index = self._search_index()
        query_vector = self._embedder.embed_query(query)
        matches = _filtered_matches(
            index,
            query_vector,
            top_k=top_k or self._default_top_k,
            filters=filters if filters is not None else self._default_filters,
        )
        return [
            _to_search_result(rank, index.feedback[position], score)
            for rank, (position, score) in enumerate(matches, start=1)
        ]

    def describe_index(self) -> SearchIndexManifest:
        return self._search_index().manifest

    def warm_up(self) -> SearchIndexManifest:
        """Load the index and the embedding model now, so the first real query is instant."""
        manifest = self.describe_index()
        self._embedder.embed_query("warm up")
        return manifest

    def _search_index(self) -> SearchIndex:
        if self._loaded is None:
            loaded = self._index_store.load()
            if loaded.manifest.model_name != self._embedder.model_name:
                raise RetrievalError(
                    f"The index was built with {loaded.manifest.model_name}, but queries would be "
                    f"embedded with {self._embedder.model_name}. Run `rag-poc build-index` to "
                    "rebuild the index with the configured model."
                )
            logger.info(
                "Loaded search index: %s vectors (%s)",
                f"{loaded.manifest.count:,}",
                loaded.manifest.index_type,
            )
            self._loaded = loaded
        return self._loaded


def _to_search_result(rank: int, item: FeedbackItem, score: float) -> SearchResult:
    return SearchResult(
        rank=rank,
        score=score,
        feedback_id=item.feedback_id,
        text=item.text,
        entity_id=item.entity_id,
        rating=item.rating,
        title=item.title,
        timestamp=item.timestamp,
    )


_INITIAL_CANDIDATES_PER_RESULT = 20


def _filtered_matches(
    index: SearchIndex, query_vector: np.ndarray, *, top_k: int, filters: SearchFilters
) -> list[tuple[int, float]]:
    """Exact top-k among items that pass ``filters``.

    Fetches a ranked candidate pool and widens it until enough candidates pass, so selective
    filters (e.g. 1-star only) still return complete, correctly ranked results.
    """
    size = index.vectors.size
    pool = min(size, max(top_k, top_k * _INITIAL_CANDIDATES_PER_RESULT))
    while True:
        candidates = index.vectors.search(query_vector, pool)
        accepted = [
            (position, score)
            for position, score in candidates
            if filters.accepts(index.feedback[position])
        ]
        if len(accepted) >= top_k or pool >= size:
            return accepted[:top_k]
        pool = min(size, pool * 4)
