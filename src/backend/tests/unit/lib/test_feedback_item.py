from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from rag_poc.lib import FeedbackItem


def _item(**overrides: object) -> FeedbackItem:
    fields: dict[str, object] = {
        "feedback_id": "fb-1",
        "source": "Yelp",
        "entity_id": "biz-9",
        "text": "Lovely staff",
    }
    return FeedbackItem(**(fields | overrides))


def test_source_is_lowercased_and_timestamps_are_normalized_to_utc() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))

    aware = _item(timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=ist))
    naive = _item(timestamp=datetime(2024, 1, 1, 10, 0))

    assert aware.source == "yelp"
    assert aware.timestamp == datetime(2024, 1, 1, 4, 30, tzinfo=UTC)
    assert naive.timestamp == datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    assert aware.metadata == {} and aware.metadata is not naive.metadata


@pytest.mark.parametrize(
    "overrides",
    [
        {"text": "   "},
        {"feedback_id": ""},
        {"rating": -1.0},
        {"rating": float("nan")},
        {"unexpected_field": 1},
    ],
)
def test_invalid_feedback_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _item(**overrides)
