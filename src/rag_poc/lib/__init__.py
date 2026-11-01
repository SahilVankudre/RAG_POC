"""Shared library: cross-module types and utilities.

Everything here may be imported by any layer, and nothing here imports other ``rag_poc``
packages.
"""

from rag_poc.lib.data_quality_report import DataQualityReport
from rag_poc.lib.dataset_reports import (
    ColumnSummary,
    DatasetProfile,
    ReviewExample,
    ReviewSampleReport,
)
from rag_poc.lib.errors import (
    ConfigurationError,
    DatasetError,
    EmbeddingError,
    EvaluationError,
    LLMError,
    RagPocError,
    RetrievalError,
)
from rag_poc.lib.evaluation_types import EvaluationQuestion, EvaluationRecord, EvaluationRun
from rag_poc.lib.feedback_adaptation import FeedbackAdaptation, RejectedRecord
from rag_poc.lib.feedback_embeddings import EmbeddingRunReport, FeedbackEmbeddings
from rag_poc.lib.feedback_item import FeedbackItem
from rag_poc.lib.feedback_item_table import feedback_items_from_frame, feedback_items_to_frame
from rag_poc.lib.llm_response import LLMResponse
from rag_poc.lib.logging_setup import configure_logging
from rag_poc.lib.pipeline_report import PipelineReport
from rag_poc.lib.prepared_feedback import PreparedFeedback
from rag_poc.lib.rag_response import AnswerStatus, InsufficiencyReason, RAGResponse, RAGSource
from rag_poc.lib.search_index_types import (
    IndexBuildReport,
    SearchFilters,
    SearchIndexManifest,
    SearchResult,
)

__all__ = [
    "AnswerStatus",
    "ColumnSummary",
    "ConfigurationError",
    "DataQualityReport",
    "DatasetError",
    "DatasetProfile",
    "EmbeddingError",
    "EmbeddingRunReport",
    "EvaluationError",
    "EvaluationQuestion",
    "EvaluationRecord",
    "EvaluationRun",
    "FeedbackAdaptation",
    "FeedbackEmbeddings",
    "FeedbackItem",
    "IndexBuildReport",
    "InsufficiencyReason",
    "LLMError",
    "LLMResponse",
    "PipelineReport",
    "PreparedFeedback",
    "RAGResponse",
    "RAGSource",
    "RagPocError",
    "RejectedRecord",
    "RetrievalError",
    "ReviewExample",
    "ReviewSampleReport",
    "SearchFilters",
    "SearchIndexManifest",
    "SearchResult",
    "configure_logging",
    "feedback_items_from_frame",
    "feedback_items_to_frame",
]
