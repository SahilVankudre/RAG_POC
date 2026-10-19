"""Use case: load a source's full raw review table and describe it (spec §2)."""

from rag_poc.ingestion import ReviewDataset, profile_reviews
from rag_poc.lib import DatasetProfile


class DatasetExploration:
    """Profiles the complete raw dataset so its fields and quality can be inspected."""

    def __init__(self, dataset: ReviewDataset) -> None:
        self._dataset = dataset

    def explore(self, *, example_count: int = 3) -> DatasetProfile:
        reviews = self._dataset.load_reviews()
        return profile_reviews(
            reviews,
            source_name=self._dataset.name,
            columns=self._dataset.columns,
            example_count=example_count,
            raw_fields=self._dataset.list_raw_fields(),
        )
