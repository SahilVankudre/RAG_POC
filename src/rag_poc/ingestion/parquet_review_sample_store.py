"""Stores the review sample as a local Parquet file."""

from pathlib import Path

import pandas as pd


class ParquetReviewSampleStore:
    """Persists the sampled review table to one Parquet file (implements ``ReviewSampleStore``)."""

    def __init__(self, location: Path) -> None:
        self._location = location

    @property
    def location(self) -> Path:
        return self._location

    def exists(self) -> bool:
        return self._location.is_file()

    def save(self, sample: pd.DataFrame) -> None:
        self._location.parent.mkdir(parents=True, exist_ok=True)
        sample.to_parquet(self._location, index=False)

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self._location)
