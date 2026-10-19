"""Stores prepared feedback as Parquet and the data-quality report as JSON."""

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from rag_poc.lib import (
    DataQualityReport,
    PreparedFeedback,
    feedback_items_from_frame,
    feedback_items_to_frame,
)

_EMBEDDING_TEXT = "embedding_text"


class ParquetProcessedFeedbackStore:
    """One row per prepared item: the ``FeedbackItem`` columns plus its embedding text.

    Implements ``ProcessedFeedbackStore``.
    """

    def __init__(self, *, location: Path, report_location: Path) -> None:
        self._location = location
        self._report_location = report_location

    @property
    def location(self) -> Path:
        return self._location

    @property
    def report_location(self) -> Path:
        return self._report_location

    def exists(self) -> bool:
        return self._location.is_file()

    def save(self, prepared: Sequence[PreparedFeedback], report: DataQualityReport) -> None:
        frame = feedback_items_to_frame([entry.item for entry in prepared])
        frame[_EMBEDDING_TEXT] = [entry.embedding_text for entry in prepared]

        self._location.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self._location, index=False)
        self._report_location.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load(self) -> list[PreparedFeedback]:
        frame = pd.read_parquet(self._location)
        items = feedback_items_from_frame(frame)
        return [
            PreparedFeedback(item=item, embedding_text=embedding_text)
            for item, embedding_text in zip(items, frame[_EMBEDDING_TEXT], strict=True)
        ]
