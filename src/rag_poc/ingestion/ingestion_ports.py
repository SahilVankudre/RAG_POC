"""Ports that any feedback source must satisfy to be explored, sampled, stored and adapted."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import pandas as pd

from rag_poc.lib import FeedbackAdaptation


@dataclass(frozen=True, slots=True)
class ReviewColumns:
    """Maps review concepts to a source's native column names, so shared logic stays generic."""

    text: str
    rating: str
    entity_id: str
    timestamp: str
    title: str | None = None
    timestamp_unit: Literal["s", "ms"] | None = None  # None: column already holds datetimes


class ReviewDataset(Protocol):
    """A source's complete raw review table, in its native columns."""

    @property
    def name(self) -> str: ...

    @property
    def columns(self) -> ReviewColumns: ...

    def load_reviews(self) -> pd.DataFrame:
        """Return every raw review; row positions must be stable between calls."""
        ...

    def list_raw_fields(self) -> tuple[str, ...]:
        """Return all fields present in the raw records, including ones not loaded."""
        ...


class ReviewSampleStore(Protocol):
    """Local persistence for the sampled review table, so it is downloaded only once."""

    @property
    def location(self) -> Path: ...

    def exists(self) -> bool: ...

    def save(self, sample: pd.DataFrame) -> None: ...

    def load(self) -> pd.DataFrame: ...


class FeedbackAdapter(Protocol):
    """Converts a source's native records into canonical ``FeedbackItem`` objects."""

    @property
    def source(self) -> str: ...

    def adapt(self, records: pd.DataFrame) -> FeedbackAdaptation:
        """Return valid items and a rejection entry for every record that could not be used."""
        ...
