"""Reproducible selection of a manageable review subset (spec §3)."""

import pandas as pd

from rag_poc.lib import DatasetError

SOURCE_ROW_COLUMN = "source_row"
"""Column added to a sample holding each review's position in the original raw table."""


def drop_reviews_without_text(reviews: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Keep only reviews whose text is present and not just whitespace."""
    text = reviews[text_column].astype("string")
    has_text = text.str.strip().ne("").fillna(False).astype(bool)
    return reviews[has_text]


def select_review_sample(
    reviews: pd.DataFrame, *, text_column: str, size: int, seed: int
) -> pd.DataFrame:
    """Draw ``size`` text-bearing reviews uniformly at random, reproducibly for a given seed.

    The original row position is preserved in ``source_row`` so downstream adapters can
    derive stable identifiers, and rows are returned in source order for readable diffs.
    """
    candidates = drop_reviews_without_text(reviews, text_column)
    if len(candidates) < size:
        raise DatasetError(
            f"Requested a sample of {size:,} reviews but only {len(candidates):,} have text."
        )
    sample = candidates.sample(n=size, random_state=seed).sort_index()
    return sample.rename_axis(SOURCE_ROW_COLUMN).reset_index()
