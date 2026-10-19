"""Ingestion module: acquires raw feedback from a source and adapts it to ``FeedbackItem``.

Ports define what a feedback source must provide; source-specific adapters (Amazon today,
other sources later) live in sub-packages and are wired only by the composition root.
"""

from rag_poc.ingestion.ingestion_ports import (
    FeedbackAdapter,
    ReviewColumns,
    ReviewDataset,
    ReviewSampleStore,
)
from rag_poc.ingestion.review_profiling import profile_reviews
from rag_poc.ingestion.review_sampling import (
    SOURCE_ROW_COLUMN,
    drop_reviews_without_text,
    select_review_sample,
)

__all__ = [
    "SOURCE_ROW_COLUMN",
    "FeedbackAdapter",
    "ReviewColumns",
    "ReviewDataset",
    "ReviewSampleStore",
    "drop_reviews_without_text",
    "profile_reviews",
    "select_review_sample",
]
