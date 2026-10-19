"""Summary statistics that help a human understand a review table (spec §2)."""

import pandas as pd

from rag_poc.ingestion.ingestion_ports import ReviewColumns
from rag_poc.ingestion.review_sampling import drop_reviews_without_text
from rag_poc.lib import ColumnSummary, DatasetProfile, ReviewExample


def profile_reviews(
    reviews: pd.DataFrame,
    *,
    source_name: str,
    columns: ReviewColumns,
    example_count: int = 3,
    seed: int = 0,
    raw_fields: tuple[str, ...] = (),
) -> DatasetProfile:
    """Describe size, columns, rating spread, time range and a few example reviews."""
    with_text = drop_reviews_without_text(reviews, columns.text)
    timestamps = _to_datetimes(reviews[columns.timestamp], columns.timestamp_unit)
    has_timestamps = bool(timestamps.notna().any())
    ratings = reviews[columns.rating].value_counts().sort_index()
    null_counts = reviews.isna().sum()

    return DatasetProfile(
        source_name=source_name,
        record_count=len(reviews),
        columns=tuple(
            ColumnSummary(name=str(name), dtype=str(dtype), null_count=int(null_counts[name]))
            for name, dtype in reviews.dtypes.items()
        ),
        reviews_without_text=len(reviews) - len(with_text),
        unique_entities=int(reviews[columns.entity_id].nunique()),
        rating_distribution={float(rating): int(count) for rating, count in ratings.items()},
        earliest_review=timestamps.min().to_pydatetime() if has_timestamps else None,
        latest_review=timestamps.max().to_pydatetime() if has_timestamps else None,
        median_text_length=_median_length(with_text[columns.text]),
        examples=_pick_examples(with_text, columns, example_count, seed),
        raw_fields=raw_fields,
    )


def _median_length(texts: pd.Series) -> float:
    return float(texts.str.len().median()) if len(texts) else 0.0


def _to_datetimes(values: pd.Series, unit: str | None) -> pd.Series:
    if unit is None:
        return pd.to_datetime(values, utc=True, errors="coerce")
    return pd.to_datetime(values, unit=unit, utc=True, errors="coerce")


def _pick_examples(
    reviews: pd.DataFrame, columns: ReviewColumns, count: int, seed: int
) -> tuple[ReviewExample, ...]:
    if count <= 0 or reviews.empty:
        return ()
    picked = reviews.sample(n=min(count, len(reviews)), random_state=seed)
    return tuple(
        ReviewExample(
            entity_id=str(row[columns.entity_id]),
            rating=None if pd.isna(row[columns.rating]) else float(row[columns.rating]),
            title=_optional_text(row, columns.title),
            text=str(row[columns.text]),
        )
        for _, row in picked.iterrows()
    )


def _optional_text(row: pd.Series, column: str | None) -> str | None:
    if column is None or pd.isna(row[column]):
        return None
    return str(row[column])
