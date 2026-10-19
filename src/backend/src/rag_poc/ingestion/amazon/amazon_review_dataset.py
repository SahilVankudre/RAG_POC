"""Fetches and reads one category of the Amazon Reviews 2023 dataset.

The Hugging Face repo still ships a legacy loading script that current ``datasets`` releases
refuse to run, so the category's raw JSONL file is downloaded directly from the Hub (cached
locally after the first run) and parsed with pyarrow.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
from huggingface_hub import hf_hub_download
from pyarrow import json as pa_json

from rag_poc.ingestion.ingestion_ports import ReviewColumns
from rag_poc.lib import DatasetError

logger = logging.getLogger(__name__)

AMAZON_REVIEW_COLUMNS = ReviewColumns(
    text="text",
    title="title",
    rating="rating",
    entity_id="asin",
    timestamp="timestamp",
    timestamp_unit="ms",
)

# Fields loaded from each record. Unlisted fields (e.g. ``images``) are skipped while parsing.
AMAZON_REVIEW_SCHEMA = pa.schema(
    [
        ("rating", pa.float64()),
        ("title", pa.string()),
        ("text", pa.string()),
        ("asin", pa.string()),
        ("parent_asin", pa.string()),
        ("user_id", pa.string()),
        ("timestamp", pa.int64()),
        ("helpful_vote", pa.int64()),
        ("verified_purchase", pa.bool_()),
    ]
)

_PARSE_BLOCK_BYTES = 16 * 1024 * 1024  # must exceed the longest single review record


def read_amazon_review_file(path: Path) -> pd.DataFrame:
    """Parse an Amazon Reviews 2023 JSONL file into a DataFrame with the loaded fields."""
    table = pa_json.read_json(
        path,
        read_options=pa_json.ReadOptions(block_size=_PARSE_BLOCK_BYTES),
        parse_options=pa_json.ParseOptions(
            explicit_schema=AMAZON_REVIEW_SCHEMA, unexpected_field_behavior="ignore"
        ),
    )
    return table.to_pandas()


class AmazonReviewDataset:
    """One Amazon Reviews 2023 category, implementing ``ReviewDataset``."""

    def __init__(self, *, repo_id: str, category: str, cache_dir: Path | None = None) -> None:
        self._repo_id = repo_id
        self._category = category
        self._cache_dir = cache_dir
        self._local_file: Path | None = None

    @property
    def name(self) -> str:
        return f"amazon/{self._category}"

    @property
    def columns(self) -> ReviewColumns:
        return AMAZON_REVIEW_COLUMNS

    def load_reviews(self) -> pd.DataFrame:
        path = self._fetch_review_file()
        logger.info("Parsing %s", path.name)
        reviews = read_amazon_review_file(path)
        logger.info("Loaded %s reviews from %s", f"{len(reviews):,}", self.name)
        return reviews

    def list_raw_fields(self) -> tuple[str, ...]:
        with self._fetch_review_file().open(encoding="utf-8") as review_file:
            return tuple(json.loads(review_file.readline()))

    def _fetch_review_file(self) -> Path:
        if self._local_file is None:
            remote_path = f"raw/review_categories/{self._category}.jsonl"
            logger.info("Fetching %s/%s (cached after first download)", self._repo_id, remote_path)
            try:
                downloaded = hf_hub_download(
                    repo_id=self._repo_id,
                    filename=remote_path,
                    repo_type="dataset",
                    cache_dir=self._cache_dir,
                )
            except Exception as error:  # network, auth or missing-file errors from the Hub
                raise DatasetError(
                    f"Could not download {remote_path} from {self._repo_id}: {error}"
                ) from error
            self._local_file = Path(downloaded)
        return self._local_file
