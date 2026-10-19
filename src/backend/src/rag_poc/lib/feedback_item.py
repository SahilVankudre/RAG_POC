"""Canonical, source-agnostic representation of one piece of customer feedback (spec §4)."""

import math
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FeedbackItem(BaseModel):
    """One piece of customer feedback, independent of where it came from.

    Amazon reviews today; Yelp reviews, survey answers or support tickets tomorrow. Every
    component after a source adapter works with this model only, never with source records.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    feedback_id: str = Field(min_length=1, description="Unique, stable id within its source.")
    source: str = Field(min_length=1, description="Lowercase origin name, e.g. 'amazon'.")
    entity_id: str = Field(
        min_length=1,
        description="Product, property or other business entity the feedback is about.",
    )
    text: str = Field(description="The feedback body, as written by the customer.")
    title: str | None = Field(default=None, description="Optional headline or subject line.")
    rating: float | None = Field(
        default=None, ge=0, description="Score on the source's own scale, if any."
    )
    timestamp: datetime | None = Field(default=None, description="When it was given, in UTC.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Source-specific extras kept for traceability."
    )

    @field_validator("source")
    @classmethod
    def _normalize_source(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("text")
    @classmethod
    def _require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value

    @field_validator("title")
    @classmethod
    def _blank_title_is_none(cls, value: str | None) -> str | None:
        return value if value and value.strip() else None

    @field_validator("rating")
    @classmethod
    def _require_finite_rating(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("rating must be a finite number")
        return value

    @field_validator("timestamp")
    @classmethod
    def _normalize_to_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:  # naive datetimes are interpreted as UTC
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
