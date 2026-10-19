"""Types describing the persisted search index and the results it returns (spec §8)."""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_poc.lib.feedback_item import FeedbackItem


@dataclass(frozen=True, slots=True)
class SearchIndexManifest:
    """Records how an index was built, so a mismatched or stale index is detected on load."""

    model_name: str
    dimension: int
    count: int
    index_type: str
    embeddings_fingerprint: str
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"created_at": self.created_at.isoformat()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchIndexManifest":
        return cls(**(data | {"created_at": datetime.fromisoformat(data["created_at"])}))


@dataclass(frozen=True, slots=True)
class IndexBuildReport:
    """Outcome of building the search index, or of reusing an up-to-date one."""

    manifest: SearchIndexManifest
    reused: bool
    seconds: float
    location: Path


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One retrieved piece of feedback, ranked by cosine similarity to the query."""

    rank: int
    score: float
    feedback_id: str
    text: str
    entity_id: str
    rating: float | None
    title: str | None
    timestamp: datetime | None


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Restricts which feedback a search may return.

    Embedding similarity matches topic, not opinion: "smell is too strong" also matches
    "smell is amazing". A rating range (e.g. ``max_rating=2`` for complaints) removes most such
    mismatches, and ``min_words`` drops very short reviews ("regular customer") that match a
    query on a single shared word. Items without a rating never pass a rating bound.
    """

    min_rating: float | None = None
    max_rating: float | None = None
    min_words: int = 0

    def __post_init__(self) -> None:
        if (
            self.min_rating is not None
            and self.max_rating is not None
            and self.min_rating > self.max_rating
        ):
            raise ValueError(f"min_rating {self.min_rating} exceeds max_rating {self.max_rating}")
        if self.min_words < 0:
            raise ValueError("min_words must not be negative")

    def accepts(self, item: FeedbackItem) -> bool:
        if self.min_words and len(item.text.split()) < self.min_words:
            return False
        if self.min_rating is None and self.max_rating is None:
            return True
        if item.rating is None:
            return False
        if self.min_rating is not None and item.rating < self.min_rating:
            return False
        return self.max_rating is None or item.rating <= self.max_rating

    def describe(self) -> str:
        """Human-readable summary, e.g. "rating <= 2, at least 3 words"."""
        parts = []
        if self.min_rating is not None and self.max_rating is not None:
            parts.append(f"rating {self.min_rating:g}-{self.max_rating:g}")
        elif self.max_rating is not None:
            parts.append(f"rating <= {self.max_rating:g}")
        elif self.min_rating is not None:
            parts.append(f"rating >= {self.min_rating:g}")
        if self.min_words:
            parts.append(f"at least {self.min_words} words")
        return ", ".join(parts) or "none"
