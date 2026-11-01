"""Summary of a full build: sample -> prepare -> embed -> index."""

from dataclasses import dataclass

from rag_poc.lib.data_quality_report import DataQualityReport
from rag_poc.lib.dataset_reports import ReviewSampleReport
from rag_poc.lib.feedback_embeddings import EmbeddingRunReport
from rag_poc.lib.search_index_types import IndexBuildReport


@dataclass(frozen=True, slots=True)
class PipelineReport:
    sample: ReviewSampleReport
    quality: DataQualityReport
    embedding: EmbeddingRunReport
    index: IndexBuildReport
    seconds: float
