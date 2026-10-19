"""Report types describing a review dataset, shared by ingestion, use cases and the CLI."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ColumnSummary:
    """One column of a loaded review table."""

    name: str
    dtype: str
    null_count: int


@dataclass(frozen=True, slots=True)
class ReviewExample:
    """A single review shown to a human for orientation."""

    entity_id: str
    rating: float | None
    title: str | None
    text: str


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Basic statistics of a review table: size, shape, coverage and a few examples."""

    source_name: str
    record_count: int
    columns: tuple[ColumnSummary, ...]
    reviews_without_text: int
    unique_entities: int
    rating_distribution: dict[float, int]
    earliest_review: datetime | None
    latest_review: datetime | None
    median_text_length: float
    examples: tuple[ReviewExample, ...]
    raw_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReviewSampleReport:
    """Outcome of preparing the local review sample."""

    location: Path
    created: bool
    profile: DatasetProfile
