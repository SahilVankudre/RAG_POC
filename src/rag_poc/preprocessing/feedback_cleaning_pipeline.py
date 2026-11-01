"""Runs the cleaning steps in order and composes embedding text for the survivors."""

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from rag_poc.lib import FeedbackItem, PreparedFeedback
from rag_poc.preprocessing.cleaning_steps import CleaningStep
from rag_poc.preprocessing.embedding_text_composition import compose_embedding_text

logger = logging.getLogger(__name__)

SHORT_TEXT_CHARS = 10
"""Texts shorter than this ("Great", "👍") are kept but counted as low-information."""

LONG_TEXT_CHARS = 1_000
"""Roughly 256 word-piece tokens: longer embedding texts are truncated by MiniLM-sized models."""


@dataclass(frozen=True, slots=True)
class CleaningOutcome:
    prepared: list[PreparedFeedback]
    exclusions: dict[str, int]
    modifications: dict[str, int]
    titles_combined: int
    short_text_records: int
    long_text_records: int


class FeedbackCleaningPipeline:
    """Applies cleaning steps in sequence, then pairs each item with its embedding text."""

    def __init__(self, steps: Sequence[CleaningStep]) -> None:
        self._steps = list(steps)

    def run(self, items: Sequence[FeedbackItem]) -> CleaningOutcome:
        exclusions: Counter[str] = Counter()
        modifications: Counter[str] = Counter()
        current = list(items)
        for step in self._steps:
            outcome = step.apply(current)
            logger.debug("%s: %d in, %d out", step.name, len(current), len(outcome.items))
            current = outcome.items
            exclusions.update(outcome.exclusions)
            modifications.update(outcome.modifications)

        prepared: list[PreparedFeedback] = []
        titles_combined = 0
        for item in current:
            embedding_text, used_title = compose_embedding_text(item)
            titles_combined += used_title
            prepared.append(PreparedFeedback(item=item, embedding_text=embedding_text))

        return CleaningOutcome(
            prepared=prepared,
            exclusions=dict(exclusions),
            modifications=dict(modifications),
            titles_combined=titles_combined,
            short_text_records=sum(len(p.item.text) < SHORT_TEXT_CHARS for p in prepared),
            long_text_records=sum(len(p.embedding_text) > LONG_TEXT_CHARS for p in prepared),
        )
