"""Builds the text that is embedded for each feedback item."""

import re

from rag_poc.lib import FeedbackItem

# Titles Amazon generated from the star rating ("Five Stars"); they add no meaning.
_GENERIC_TITLE = re.compile(r"^(?:one|two|three|four|five|[1-5])\s+stars?[.!]*$", re.IGNORECASE)
_TRAILING_PUNCTUATION = ".!?…:;"


def compose_embedding_text(item: FeedbackItem) -> tuple[str, bool]:
    """Return the embedding text and whether the title was combined into it.

    A title is prepended only when it adds information: not missing, not auto-generated
    from the rating, and not already the start of the body.
    """
    title = item.title
    if not title or not _title_adds_information(title, item.text):
        return item.text, False
    separator = " " if title[-1] in _TRAILING_PUNCTUATION else ". "
    return f"{title}{separator}{item.text}", True


def is_generic_title(title: str) -> bool:
    """True for titles generated from the star rating, e.g. "Five Stars"."""
    return bool(_GENERIC_TITLE.match(title.strip()))


def _title_adds_information(title: str, text: str) -> bool:
    if is_generic_title(title):
        return False
    core = title.rstrip(_TRAILING_PUNCTUATION + " ").casefold()
    return bool(core) and not text.casefold().startswith(core)
