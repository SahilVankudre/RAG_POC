"""Converts Amazon Reviews 2023 records into canonical ``FeedbackItem`` objects (spec §5).

Field mapping::

    asin                        -> entity_id   (parent_asin if asin is missing)
    text                        -> text
    title                       -> title
    rating                      -> rating      (1-5; anything else becomes None)
    timestamp (epoch ms)        -> timestamp   (UTC datetime)
    "amazon"                    -> source
    parent_asin, user_id, verified_purchase, helpful_vote, source_row, category -> metadata

Amazon records carry no review id, so ``feedback_id`` is derived from the review's natural key
(user_id, asin, timestamp). The same review therefore gets the same id on every run regardless
of sampling or row order. Records missing part of that key fall back to their original row
position in the raw file.
"""

import hashlib
import logging
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from pydantic import ValidationError

from rag_poc.ingestion.review_sampling import SOURCE_ROW_COLUMN
from rag_poc.lib import FeedbackAdaptation, FeedbackItem, RejectedRecord

logger = logging.getLogger(__name__)

AMAZON_SOURCE = "amazon"
FEEDBACK_ID_PREFIX = "amz_"

MISSING_TEXT = "missing_text"
MISSING_ENTITY_ID = "missing_entity_id"
DUPLICATE_FEEDBACK_ID = "duplicate_feedback_id"
INVALID_RECORD = "invalid_record"

_RATING_RANGE = (1.0, 5.0)


def build_amazon_feedback_id(
    *, user_id: str | None, asin: str | None, timestamp_ms: int | None, position: int
) -> str:
    """Stable id from the review's natural key, or from its raw-file position as a fallback."""
    if user_id and asin and timestamp_ms is not None:
        natural_key = f"{user_id}|{asin}|{timestamp_ms}"
    else:
        natural_key = f"row|{position}"
    digest = hashlib.sha256(natural_key.encode("utf-8")).hexdigest()
    return f"{FEEDBACK_ID_PREFIX}{digest[:16]}"


class AmazonReviewAdapter:
    """Turns Amazon review rows into valid ``FeedbackItem`` objects (a ``FeedbackAdapter``)."""

    def __init__(self, *, category: str) -> None:
        self._category = category

    @property
    def source(self) -> str:
        return AMAZON_SOURCE

    def adapt(self, records: pd.DataFrame) -> FeedbackAdaptation:
        adaptation = FeedbackAdaptation()
        seen_ids: set[str] = set()

        for offset, record in enumerate(records.to_dict("records")):
            position = _as_int(record.get(SOURCE_ROW_COLUMN))
            position = offset if position is None else position
            outcome = self._to_feedback_item(record, position)

            if isinstance(outcome, RejectedRecord):
                adaptation.rejections.append(outcome)
            elif outcome.feedback_id in seen_ids:
                adaptation.rejections.append(
                    RejectedRecord(position, DUPLICATE_FEEDBACK_ID, outcome.feedback_id)
                )
            else:
                seen_ids.add(outcome.feedback_id)
                adaptation.items.append(outcome)

        logger.info(
            "Adapted %s Amazon reviews; rejected %s %s",
            f"{len(adaptation.items):,}",
            f"{len(adaptation.rejections):,}",
            adaptation.rejection_counts or "",
        )
        return adaptation

    def _to_feedback_item(
        self, record: Mapping[str, Any], position: int
    ) -> FeedbackItem | RejectedRecord:
        text = _as_text(record.get("text"))
        if text is None:
            return RejectedRecord(position, MISSING_TEXT)

        asin = _as_text(record.get("asin"))
        parent_asin = _as_text(record.get("parent_asin"))
        entity_id = asin or parent_asin
        if entity_id is None:
            return RejectedRecord(position, MISSING_ENTITY_ID)

        user_id = _as_text(record.get("user_id"))
        timestamp_ms = _as_int(record.get("timestamp"))
        metadata = {
            "category": self._category,
            "parent_asin": parent_asin,
            "user_id": user_id,
            "verified_purchase": _as_bool(record.get("verified_purchase")),
            "helpful_vote": _as_int(record.get("helpful_vote")),
            SOURCE_ROW_COLUMN: position,
        }

        try:
            return FeedbackItem(
                feedback_id=build_amazon_feedback_id(
                    user_id=user_id, asin=entity_id, timestamp_ms=timestamp_ms, position=position
                ),
                source=AMAZON_SOURCE,
                entity_id=entity_id,
                text=text,
                title=_as_text(record.get("title")),
                rating=_as_rating(record.get("rating")),
                timestamp=_as_utc_datetime(timestamp_ms),
                metadata={key: value for key, value in metadata.items() if value is not None},
            )
        except ValidationError as error:
            first = error.errors()[0]
            detail = f"{'.'.join(str(part) for part in first['loc'])}: {first['msg']}"
            return RejectedRecord(position, INVALID_RECORD, detail)


def _is_missing(value: Any) -> bool:
    return value is None or (not isinstance(value, (list, dict)) and bool(pd.isna(value)))


def _as_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value)
    return text if text.strip() else None


def _as_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    return None if _is_missing(value) else bool(value)


def _as_rating(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    low, high = _RATING_RANGE
    return rating if math.isfinite(rating) and low <= rating <= high else None


def _as_utc_datetime(timestamp_ms: int | None) -> datetime | None:
    if timestamp_ms is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None
