"""Use case: embed every prepared feedback item and store the vectors (spec §7)."""

import logging
import time

from rag_poc.embedding import (
    Embedder,
    EmbeddingStore,
    ProgressCallback,
    embed_in_batches,
    fingerprint_embedding_input,
)
from rag_poc.lib import DatasetError, EmbeddingRunReport, FeedbackEmbeddings
from rag_poc.preprocessing import ProcessedFeedbackStore

logger = logging.getLogger(__name__)


class FeedbackEmbedding:
    """Embeds prepared feedback in batches, skipping the work when nothing has changed."""

    def __init__(
        self,
        *,
        processed_store: ProcessedFeedbackStore,
        embedder: Embedder,
        embedding_store: EmbeddingStore,
        batch_size: int,
    ) -> None:
        self._processed_store = processed_store
        self._embedder = embedder
        self._embedding_store = embedding_store
        self._batch_size = batch_size

    def embed_feedback(
        self, *, force: bool = False, on_progress: ProgressCallback | None = None
    ) -> EmbeddingRunReport:
        """Embed all prepared feedback, or reuse stored vectors built from identical input."""
        if not self._processed_store.exists():
            raise DatasetError(
                f"No prepared feedback at {self._processed_store.location}. "
                "Run `rag-poc prepare` first."
            )
        prepared = self._processed_store.load()
        feedback_ids = [entry.feedback_id for entry in prepared]
        texts = [entry.embedding_text for entry in prepared]
        fingerprint = fingerprint_embedding_input(self._embedder.model_name, feedback_ids, texts)

        if not force and self._embedding_store.stored_fingerprint() == fingerprint:
            stored = self._embedding_store.load()
            logger.info("Embeddings are up to date; reusing %s", self._embedding_store.location)
            return self._report(stored, seconds=0.0, reused=True)

        started = time.perf_counter()
        vectors = embed_in_batches(
            self._embedder, texts, batch_size=self._batch_size, on_progress=on_progress
        )
        seconds = time.perf_counter() - started
        embeddings = FeedbackEmbeddings(
            feedback_ids=tuple(feedback_ids),
            vectors=vectors,
            model_name=self._embedder.model_name,
        )
        self._embedding_store.save(embeddings, fingerprint=fingerprint)
        logger.info(
            "Embedded %s texts in %.1fs (%.0f/s)",
            f"{embeddings.count:,}",
            seconds,
            embeddings.count / seconds if seconds else 0,
        )
        return self._report(embeddings, seconds=seconds, reused=False)

    def _report(
        self, embeddings: FeedbackEmbeddings, *, seconds: float, reused: bool
    ) -> EmbeddingRunReport:
        return EmbeddingRunReport(
            model_name=embeddings.model_name,
            count=embeddings.count,
            dimension=embeddings.dimension,
            batch_size=self._batch_size,
            seconds=seconds,
            reused=reused,
            location=self._embedding_store.location,
        )
