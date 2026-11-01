"""Converts ``FeedbackItem`` collections to and from DataFrames for columnar storage.

``metadata`` is stored as a JSON string so the table schema stays stable regardless of which
source-specific keys an item carries; timestamps are stored as UTC datetimes.
"""

import json
from collections.abc import Sequence

import pandas as pd

from rag_poc.lib.feedback_item import FeedbackItem

FEEDBACK_ITEM_COLUMNS = tuple(FeedbackItem.model_fields)


def feedback_items_to_frame(items: Sequence[FeedbackItem]) -> pd.DataFrame:
    """One row per item, in the given order, with the columns of ``FeedbackItem``."""
    rows = [
        item.model_dump() | {"metadata": json.dumps(item.metadata, ensure_ascii=False)}
        for item in items
    ]
    frame = pd.DataFrame(rows, columns=list(FEEDBACK_ITEM_COLUMNS))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def feedback_items_from_frame(frame: pd.DataFrame) -> list[FeedbackItem]:
    """Rebuild validated items from the ``FeedbackItem`` columns of ``frame``, in row order."""
    columns = frame[list(FEEDBACK_ITEM_COLUMNS)]
    columns = columns.astype(object).where(columns.notna(), None)
    items = []
    for row in columns.to_dict("records"):
        row["metadata"] = json.loads(row["metadata"])
        if row["timestamp"] is not None:
            row["timestamp"] = row["timestamp"].to_pydatetime()
        items.append(FeedbackItem(**row))
    return items
