"""Amazon Reviews 2023 source adapter (McAuley Lab, hosted on Hugging Face)."""

from rag_poc.ingestion.amazon.amazon_review_adapter import (
    AMAZON_SOURCE,
    AmazonReviewAdapter,
    build_amazon_feedback_id,
)
from rag_poc.ingestion.amazon.amazon_review_dataset import (
    AMAZON_REVIEW_COLUMNS,
    AmazonReviewDataset,
    read_amazon_review_file,
)

__all__ = [
    "AMAZON_REVIEW_COLUMNS",
    "AMAZON_SOURCE",
    "AmazonReviewAdapter",
    "AmazonReviewDataset",
    "build_amazon_feedback_id",
    "read_amazon_review_file",
]
