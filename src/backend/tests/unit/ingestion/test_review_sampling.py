import pandas as pd
import pytest

from rag_poc.ingestion import SOURCE_ROW_COLUMN, drop_reviews_without_text, select_review_sample
from rag_poc.lib import DatasetError


def _reviews(texts: list[str | None]) -> pd.DataFrame:
    return pd.DataFrame({"text": texts, "rating": [5.0] * len(texts)})


def test_reviews_with_missing_or_blank_text_are_dropped() -> None:
    reviews = _reviews(["Great serum", None, "", "   \n", "Broke after a week"])

    kept = drop_reviews_without_text(reviews, "text")

    assert kept["text"].tolist() == ["Great serum", "Broke after a week"]


def test_sample_is_reproducible_and_records_source_rows() -> None:
    reviews = _reviews([f"review {i}" if i % 3 else "" for i in range(30)])

    first = select_review_sample(reviews, text_column="text", size=8, seed=7)
    second = select_review_sample(reviews, text_column="text", size=8, seed=7)

    pd.testing.assert_frame_equal(first, second)
    rows = first[SOURCE_ROW_COLUMN].tolist()
    assert rows == sorted(rows)
    assert all(first["text"] == [f"review {row}" for row in rows])


def test_sample_larger_than_available_text_reviews_is_rejected() -> None:
    reviews = _reviews(["one", "", "two"])

    with pytest.raises(DatasetError, match="only 2 have text"):
        select_review_sample(reviews, text_column="text", size=3, seed=0)
