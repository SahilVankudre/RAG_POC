"""Preprocessing module: cleans and validates ``FeedbackItem`` records for embedding.

Produces embedding-ready text plus a data-quality report of excluded records and reasons.
"""

from rag_poc.preprocessing.cleaning_steps import (
    BLANK_AFTER_CLEANING,
    DUPLICATE_TEXT,
    CleaningStep,
    DuplicateFeedbackFilter,
    ReviewTextNormalization,
    StepOutcome,
)
from rag_poc.preprocessing.embedding_text_composition import compose_embedding_text
from rag_poc.preprocessing.feedback_cleaning_pipeline import (
    CleaningOutcome,
    FeedbackCleaningPipeline,
)
from rag_poc.preprocessing.preprocessing_ports import ProcessedFeedbackStore

__all__ = [
    "BLANK_AFTER_CLEANING",
    "DUPLICATE_TEXT",
    "CleaningOutcome",
    "CleaningStep",
    "DuplicateFeedbackFilter",
    "FeedbackCleaningPipeline",
    "ProcessedFeedbackStore",
    "ReviewTextNormalization",
    "StepOutcome",
    "compose_embedding_text",
]
