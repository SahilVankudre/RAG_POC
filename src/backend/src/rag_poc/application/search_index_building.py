"""Use case: build the searchable vector index from stored embeddings (spec §8)."""

import logging
import time
from datetime import UTC, datetime

from rag_poc.embedding import EmbeddingStore, fingerprint_embedding_input
from rag_poc.lib import DatasetError, IndexBuildReport, RetrievalError, SearchIndexManifest
from rag_poc.preprocessing import ProcessedFeedbackStore
from rag_poc.retrieval import SearchIndex, SearchIndexStore, VectorIndexFactory

logger = logging.getLogger(__name__)


class SearchIndexBuilding:
    """Indexes stored embeddings and maps each vector position back to its ``FeedbackItem``.

    Needs no embedding model: vectors come from the embedding store, so building (or
    rebuilding) the index never re-embeds anything.
    """

    def __init__(
        self,
        *,
        embedding_store: EmbeddingStore,
        processed_store: ProcessedFeedbackStore,
        index_store: SearchIndexStore,
        index_factory: VectorIndexFactory,
    ) -> None:
        self._embedding_store = embedding_store
        self._processed_store = processed_store
        self._index_store = index_store
        self._index_factory = index_factory

    def build_index(self, *, force: bool = False) -> IndexBuildReport:
        """Build and save the index, or reuse the stored one if built from the same embeddings."""
        fingerprint = self._embedding_store.stored_fingerprint()
        if fingerprint is None:
            raise DatasetError(
                f"No embeddings at {self._embedding_store.location}. Run `rag-poc embed` first."
            )
        existing = self._index_store.read_manifest()
        if not force and existing and existing.embeddings_fingerprint == fingerprint:
            logger.info("Search index is up to date; reusing %s", self._index_store.location)
            return IndexBuildReport(
                manifest=existing, reused=True, seconds=0.0, location=self._index_store.location
            )

        started = time.perf_counter()
        embeddings = self._embedding_store.load()
        prepared = self._processed_store.load()
        current = fingerprint_embedding_input(
            embeddings.model_name,
            [entry.feedback_id for entry in prepared],
            [entry.embedding_text for entry in prepared],
        )
        if current != fingerprint:
            raise RetrievalError(
                "Stored embeddings were computed from different prepared feedback. "
                "Run `rag-poc embed` to refresh them before indexing."
            )

        items_by_id = {entry.feedback_id: entry.item for entry in prepared}
        feedback = [items_by_id[feedback_id] for feedback_id in embeddings.feedback_ids]
        vectors = self._index_factory(embeddings.vectors)
        manifest = SearchIndexManifest(
            model_name=embeddings.model_name,
            dimension=vectors.dimension,
            count=vectors.size,
            index_type=vectors.index_type,
            embeddings_fingerprint=fingerprint,
            created_at=datetime.now(UTC),
        )
        self._index_store.save(SearchIndex(vectors=vectors, feedback=feedback, manifest=manifest))
        seconds = time.perf_counter() - started
        logger.info("Indexed %s vectors in %.2fs", f"{manifest.count:,}", seconds)
        return IndexBuildReport(
            manifest=manifest, reused=False, seconds=seconds, location=self._index_store.location
        )
