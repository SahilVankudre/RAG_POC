"""Use case: draw a reproducible review sample and keep it locally (spec §3)."""

import logging

from rag_poc.ingestion import (
    ReviewDataset,
    ReviewSampleStore,
    profile_reviews,
    select_review_sample,
)
from rag_poc.lib import ReviewSampleReport

logger = logging.getLogger(__name__)


class ReviewSampleCreation:
    """Creates the working sample once; later runs reuse it instead of re-downloading."""

    def __init__(
        self,
        *,
        dataset: ReviewDataset,
        store: ReviewSampleStore,
        sample_size: int,
        seed: int,
    ) -> None:
        self._dataset = dataset
        self._store = store
        self._sample_size = sample_size
        self._seed = seed

    def create_sample(self, *, force: bool = False, example_count: int = 3) -> ReviewSampleReport:
        """Create (or with ``force``, recreate) the sample and return a profile of it."""
        created = force or not self._store.exists()
        if created:
            reviews = self._dataset.load_reviews()
            sample = select_review_sample(
                reviews,
                text_column=self._dataset.columns.text,
                size=self._sample_size,
                seed=self._seed,
            )
            self._store.save(sample)
            logger.info("Saved %s sampled reviews to %s", f"{len(sample):,}", self._store.location)
        else:
            sample = self._store.load()
            logger.info("Reusing existing sample at %s", self._store.location)

        profile = profile_reviews(
            sample,
            source_name=f"{self._dataset.name} (sample, seed={self._seed})",
            columns=self._dataset.columns,
            example_count=example_count,
            seed=self._seed,
        )
        return ReviewSampleReport(location=self._store.location, created=created, profile=profile)
