"""Use case: run every stage from raw reviews to a searchable index in one step."""

import logging
import time

from rag_poc.application.feedback_embedding import FeedbackEmbedding
from rag_poc.application.feedback_preparation import FeedbackPreparation
from rag_poc.application.review_sample_creation import ReviewSampleCreation
from rag_poc.application.search_index_building import SearchIndexBuilding
from rag_poc.embedding import ProgressCallback
from rag_poc.lib import PipelineReport

logger = logging.getLogger(__name__)


class SearchIndexPipeline:
    """Sample -> prepare -> embed -> index, reusing every stage whose input is unchanged.

    The sample is reused if present (it is seeded, so it never changes), preparation always
    re-runs (fast and deterministic), and embedding and indexing are skipped when their content
    fingerprints match. ``force`` recomputes embeddings and the index.
    """

    def __init__(
        self,
        *,
        sample_creation: ReviewSampleCreation,
        preparation: FeedbackPreparation,
        embedding: FeedbackEmbedding,
        index_building: SearchIndexBuilding,
    ) -> None:
        self._sample_creation = sample_creation
        self._preparation = preparation
        self._embedding = embedding
        self._index_building = index_building

    def run(
        self, *, force: bool = False, on_embedding_progress: ProgressCallback | None = None
    ) -> PipelineReport:
        started = time.perf_counter()
        logger.info("Stage 1/4: review sample")
        sample = self._sample_creation.create_sample(example_count=0)
        logger.info("Stage 2/4: cleaning and preparation")
        quality = self._preparation.prepare()
        logger.info("Stage 3/4: embeddings")
        embedding = self._embedding.embed_feedback(force=force, on_progress=on_embedding_progress)
        logger.info("Stage 4/4: search index")
        index = self._index_building.build_index(force=force)
        return PipelineReport(
            sample=sample,
            quality=quality,
            embedding=embedding,
            index=index,
            seconds=time.perf_counter() - started,
        )
