"""Composition root: the single place where concrete adapters are constructed and wired.

Feature modules and use cases only know about ports (Protocols). The container decides which
implementation backs each port, so swapping Amazon for another source, or FAISS for another
vector store, is a change here and nowhere else.

Dependencies are exposed as ``cached_property`` members: each is built on first access and
shared afterwards, so heavy collaborators (embedding model, FAISS index) load only for commands
that actually need them. Adapters with costly imports are imported inside their factory for the
same reason.
"""

from functools import cached_property

from rag_poc.application.dataset_exploration import DatasetExploration
from rag_poc.application.feedback_embedding import FeedbackEmbedding
from rag_poc.application.feedback_preparation import FeedbackPreparation
from rag_poc.application.feedback_search import FeedbackSearch
from rag_poc.application.review_sample_creation import ReviewSampleCreation
from rag_poc.application.search_index_building import SearchIndexBuilding
from rag_poc.application.search_index_pipeline import SearchIndexPipeline
from rag_poc.bootstrap.settings import AppSettings, load_settings
from rag_poc.embedding import Embedder, EmbeddingStore
from rag_poc.embedding.numpy_embedding_store import NumpyEmbeddingStore
from rag_poc.ingestion import FeedbackAdapter, ReviewDataset, ReviewSampleStore
from rag_poc.ingestion.parquet_review_sample_store import ParquetReviewSampleStore
from rag_poc.lib import SearchFilters
from rag_poc.preprocessing import (
    DuplicateFeedbackFilter,
    FeedbackCleaningPipeline,
    ProcessedFeedbackStore,
    ReviewTextNormalization,
)
from rag_poc.preprocessing.parquet_processed_feedback_store import ParquetProcessedFeedbackStore
from rag_poc.retrieval import SearchIndexStore, VectorIndexFactory


class AppContainer:
    """Owns the application settings and lazily builds the object graph from them."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    @property
    def settings(self) -> AppSettings:
        return self._settings

    # --- Ingestion adapters -------------------------------------------------------------

    @cached_property
    def review_dataset(self) -> ReviewDataset:
        from rag_poc.ingestion.amazon import AmazonReviewDataset

        return AmazonReviewDataset(
            repo_id=self._settings.hf_dataset_repo,
            category=self._settings.amazon_category,
            cache_dir=self._settings.hf_cache_dir,
        )

    @cached_property
    def review_sample_store(self) -> ReviewSampleStore:
        return ParquetReviewSampleStore(self._settings.review_sample_path)

    @cached_property
    def feedback_adapter(self) -> FeedbackAdapter:
        from rag_poc.ingestion.amazon import AmazonReviewAdapter

        return AmazonReviewAdapter(category=self._settings.amazon_category)

    # --- Preprocessing ------------------------------------------------------------------

    @cached_property
    def cleaning_pipeline(self) -> FeedbackCleaningPipeline:
        return FeedbackCleaningPipeline(
            [
                ReviewTextNormalization(),
                DuplicateFeedbackFilter(author_metadata_key="user_id"),
            ]
        )

    @cached_property
    def processed_feedback_store(self) -> ProcessedFeedbackStore:
        return ParquetProcessedFeedbackStore(
            location=self._settings.prepared_feedback_path,
            report_location=self._settings.quality_report_path,
        )

    # --- Embedding ----------------------------------------------------------------------

    @cached_property
    def embedder(self) -> Embedder:
        from rag_poc.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            self._settings.embedding_model_name,
            query_prefix=self._settings.embedding_query_prefix,
        )

    @cached_property
    def embedding_store(self) -> EmbeddingStore:
        return NumpyEmbeddingStore(
            vectors_location=self._settings.embeddings_path,
            manifest_location=self._settings.embeddings_manifest_path,
        )

    # --- Retrieval ----------------------------------------------------------------------

    @cached_property
    def vector_index_factory(self) -> VectorIndexFactory:
        from rag_poc.retrieval.faiss_vector_index import FaissVectorIndex

        return FaissVectorIndex.from_vectors

    @cached_property
    def search_index_store(self) -> SearchIndexStore:
        from rag_poc.retrieval.faiss_search_index_store import FaissSearchIndexStore

        return FaissSearchIndexStore(
            index_location=self._settings.faiss_index_path,
            id_map_location=self._settings.index_id_map_path,
            manifest_location=self._settings.index_manifest_path,
        )

    # --- Use cases ----------------------------------------------------------------------

    @cached_property
    def dataset_exploration(self) -> DatasetExploration:
        return DatasetExploration(self.review_dataset)

    @cached_property
    def review_sample_creation(self) -> ReviewSampleCreation:
        return ReviewSampleCreation(
            dataset=self.review_dataset,
            store=self.review_sample_store,
            sample_size=self._settings.sample_size,
            seed=self._settings.random_seed,
        )

    @cached_property
    def feedback_preparation(self) -> FeedbackPreparation:
        return FeedbackPreparation(
            sample_store=self.review_sample_store,
            adapter=self.feedback_adapter,
            pipeline=self.cleaning_pipeline,
            processed_store=self.processed_feedback_store,
        )

    @cached_property
    def feedback_embedding(self) -> FeedbackEmbedding:
        return FeedbackEmbedding(
            processed_store=self.processed_feedback_store,
            embedder=self.embedder,
            embedding_store=self.embedding_store,
            batch_size=self._settings.embedding_batch_size,
        )

    @cached_property
    def search_index_building(self) -> SearchIndexBuilding:
        return SearchIndexBuilding(
            embedding_store=self.embedding_store,
            processed_store=self.processed_feedback_store,
            index_store=self.search_index_store,
            index_factory=self.vector_index_factory,
        )

    @cached_property
    def search_index_pipeline(self) -> SearchIndexPipeline:
        return SearchIndexPipeline(
            sample_creation=self.review_sample_creation,
            preparation=self.feedback_preparation,
            embedding=self.feedback_embedding,
            index_building=self.search_index_building,
        )

    @cached_property
    def feedback_search(self) -> FeedbackSearch:
        return FeedbackSearch(
            index_store=self.search_index_store,
            embedder=self.embedder,
            default_top_k=self._settings.default_top_k,
            default_filters=SearchFilters(min_words=self._settings.search_min_words),
        )


def build_container(settings: AppSettings | None = None) -> AppContainer:
    """Create the application container; loads settings from the environment if not given."""
    return AppContainer(settings if settings is not None else load_settings())
