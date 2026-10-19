"""End to end: raw reviews -> sample -> prepare -> embed -> FAISS index -> search, real model.

Only the Hugging Face download is replaced (by an in-memory review table); everything else is
the production object graph from the composition root. Run with: pytest -m integration.
"""

from functools import cached_property
from pathlib import Path

import pandas as pd
import pytest

from rag_poc.bootstrap import AppContainer, load_settings
from rag_poc.ingestion import ReviewColumns, ReviewDataset
from rag_poc.ingestion.amazon import AMAZON_REVIEW_COLUMNS
from rag_poc.lib import SearchFilters

pytestmark = pytest.mark.integration

REVIEWS = [
    (1.0, "The battery stopped holding a charge after two weeks."),
    (2.0, "Charger cable broke and the device will not charge anymore."),
    (5.0, "Lovely floral scent that lasts all day."),
    (1.0, "The bottle arrived cracked and leaked everywhere."),
    (1.0, "My skin broke out in a rash after one use."),
    (5.0, "Great value, works exactly as described."),
    (5.0, "Battery life is amazing and it charges quickly."),
    (4.0, "Gentle shampoo that leaves hair soft and shiny."),
    (3.0, "   "),  # no text: dropped when sampling
]


class InMemoryAmazonDataset:
    name = "amazon/test"

    @property
    def columns(self) -> ReviewColumns:
        return AMAZON_REVIEW_COLUMNS

    def load_reviews(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "rating": [rating for rating, _ in REVIEWS],
                "title": [""] * len(REVIEWS),
                "text": [text for _, text in REVIEWS],
                "asin": [f"B{i:03d}" for i in range(len(REVIEWS))],
                "parent_asin": [f"P{i:03d}" for i in range(len(REVIEWS))],
                "user_id": [f"U{i}" for i in range(len(REVIEWS))],
                "timestamp": [1_600_000_000_000 + i for i in range(len(REVIEWS))],
                "helpful_vote": [0] * len(REVIEWS),
                "verified_purchase": [True] * len(REVIEWS),
            }
        )

    def list_raw_fields(self) -> tuple[str, ...]:
        return tuple(self.load_reviews().columns)


class OfflineContainer(AppContainer):
    @cached_property
    def review_dataset(self) -> ReviewDataset:
        return InMemoryAmazonDataset()


def test_build_then_search_from_a_fresh_container(tmp_path: Path) -> None:
    settings = load_settings(_env_file=None, data_dir=tmp_path, sample_size=8, search_min_words=0)

    first = OfflineContainer(settings).search_index_pipeline.run()
    assert first.quality.prepared_records == 8 and first.quality.is_reconciled
    assert not first.embedding.reused and first.index.manifest.count == 8

    rerun = OfflineContainer(settings).search_index_pipeline.run()
    assert rerun.embedding.reused and rerun.index.reused  # reloaded, nothing re-embedded

    search = OfflineContainer(settings).feedback_search  # fresh process: index loaded from disk
    top = search.search("device won't charge", top_k=3)
    complaints = search.search("device won't charge", top_k=2, filters=SearchFilters(max_rating=2))

    assert "charge" in top[0].text and top[0].score > top[-1].score
    assert {result.text for result in complaints} == {REVIEWS[0][1], REVIEWS[1][1]}
