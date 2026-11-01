"""Outcome of converting source records into ``FeedbackItem`` objects."""

from collections import Counter
from dataclasses import dataclass, field

from rag_poc.lib.feedback_item import FeedbackItem


@dataclass(frozen=True, slots=True)
class RejectedRecord:
    """A source record that could not become a valid ``FeedbackItem``, and why."""

    position: int
    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class FeedbackAdaptation:
    """Valid items plus every rejected record, so nothing is dropped silently."""

    items: list[FeedbackItem] = field(default_factory=list)
    rejections: list[RejectedRecord] = field(default_factory=list)

    @property
    def rejection_counts(self) -> dict[str, int]:
        """Number of rejected records per reason."""
        return dict(Counter(rejection.reason for rejection in self.rejections))
