from datetime import UTC, datetime

import pandas as pd

from rag_poc.ingestion.amazon import AmazonReviewAdapter
from rag_poc.ingestion.amazon.amazon_review_adapter import (
    DUPLICATE_FEEDBACK_ID,
    MISSING_ENTITY_ID,
    MISSING_TEXT,
)

REVIEW = {
    "source_row": 41,
    "rating": 2.0,
    "title": "Snapped",
    "text": "The handle broke after a week.",
    "asin": "B0001",
    "parent_asin": "P0001",
    "user_id": "U1",
    "timestamp": 1588687728923,
    "helpful_vote": 3,
    "verified_purchase": True,
}


def _adapt(*records: dict[str, object]):
    return AmazonReviewAdapter(category="All_Beauty").adapt(pd.DataFrame(list(records)))


def test_amazon_fields_map_onto_feedback_item() -> None:
    [item] = _adapt(REVIEW).items

    assert item.source == "amazon"
    assert item.feedback_id.startswith("amz_")
    assert item.entity_id == "B0001"
    assert (item.text, item.title, item.rating) == (REVIEW["text"], "Snapped", 2.0)
    assert item.timestamp == datetime(2020, 5, 5, 14, 8, 48, 923000, tzinfo=UTC)
    assert item.metadata == {
        "category": "All_Beauty",
        "parent_asin": "P0001",
        "user_id": "U1",
        "verified_purchase": True,
        "helpful_vote": 3,
        "source_row": 41,
    }


def test_feedback_id_is_stable_and_falls_back_to_row_position() -> None:
    moved = REVIEW | {"source_row": 999}
    anonymous = REVIEW | {"user_id": None}

    first = _adapt(REVIEW).items[0].feedback_id
    assert _adapt(moved).items[0].feedback_id == first  # natural key, not position
    assert _adapt(anonymous).items[0].feedback_id != first  # falls back to row position
    assert _adapt(anonymous).items[0].feedback_id == _adapt(anonymous).items[0].feedback_id


def test_unusable_records_are_rejected_and_missing_optionals_tolerated() -> None:
    sparse = {"source_row": 7, "text": "Fine", "asin": None, "parent_asin": "P7"}
    adaptation = _adapt(
        REVIEW,
        REVIEW | {"source_row": 42},  # same natural key -> duplicate
        REVIEW | {"source_row": 43, "text": "  "},
        REVIEW | {"source_row": 44, "asin": None, "parent_asin": None, "user_id": "U2"},
        sparse,
    )

    assert adaptation.rejection_counts == {
        DUPLICATE_FEEDBACK_ID: 1,
        MISSING_TEXT: 1,
        MISSING_ENTITY_ID: 1,
    }
    fallback = adaptation.items[-1]
    assert fallback.entity_id == "P7"
    assert fallback.rating is None and fallback.timestamp is None and fallback.title is None
