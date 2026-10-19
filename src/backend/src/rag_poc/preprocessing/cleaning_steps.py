"""Composable cleaning steps applied to ``FeedbackItem`` collections (spec §6).

Cleaning is deliberately light: modern embedding models benefit from natural language, so
stopwords, casing, punctuation and emoji are kept. Steps only remove markup noise, tidy
whitespace and drop records that carry no usable text or repeat one already kept.
"""

import html
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rag_poc.lib import FeedbackItem
from rag_poc.preprocessing.embedding_text_composition import is_generic_title

BLANK_AFTER_CLEANING = "blank_after_cleaning"
DUPLICATE_TEXT = "duplicate_text"


@dataclass(slots=True)
class StepOutcome:
    """Items kept by a step, plus counts of what it excluded and modified."""

    items: list[FeedbackItem]
    exclusions: Counter[str] = field(default_factory=Counter)
    modifications: Counter[str] = field(default_factory=Counter)


class CleaningStep(Protocol):
    """One stage of the cleaning pipeline."""

    @property
    def name(self) -> str: ...

    def apply(self, items: Sequence[FeedbackItem]) -> StepOutcome: ...


# Line-break and simple formatting tags that reviews contain as literal text (e.g. "<br />").
_HTML_TAG = re.compile(r"</?(?:br|p|div|span|b|i|u|em|strong)\b[^>]*>", re.IGNORECASE)
# Placeholders where the review had an embedded video, e.g. "[[VIDEOID:72f7bd37...]]".
_VIDEO_MARKER = re.compile(r"\[\[VIDEOID:[^\]]*\]\]", re.IGNORECASE)
_HTML_ENTITY = re.compile(r"&(?:#\d+|#x[0-9a-f]+|[a-z]+);", re.IGNORECASE)
_INVISIBLE = re.compile("[​﻿]")  # zero-width space, BOM (ZWJ kept: emoji use it)
_WHITESPACE_RUN = re.compile(r"\s+")


class ReviewTextNormalization:
    """Strips markup noise and normalizes whitespace in text and title.

    Originals are preserved in ``metadata`` (``original_text`` / ``original_title``) whenever
    a value changes. When only markup remains of the body (e.g. a video-only review), an
    informative title becomes the text; items with no usable text at all are excluded.
    """

    name = "text_normalization"

    def apply(self, items: Sequence[FeedbackItem]) -> StepOutcome:
        outcome = StepOutcome(items=[])
        for item in items:
            text = self._normalize(item.text, outcome.modifications)
            title = self._normalize(item.title, Counter()) if item.title else None
            if not text and title and not is_generic_title(title):
                text = title
                outcome.modifications["title_used_as_text"] += 1
            if not text:
                outcome.exclusions[BLANK_AFTER_CLEANING] += 1
                continue
            outcome.items.append(_with_cleaned_values(item, text=text, title=title or None))
        return outcome

    @staticmethod
    def _normalize(value: str, modifications: Counter[str]) -> str:
        cleaned = value
        for kind, pattern, replacement in (
            ("html_tags_removed", _HTML_TAG, " "),
            ("video_markers_removed", _VIDEO_MARKER, " "),
        ):
            cleaned, replaced = pattern.subn(replacement, cleaned)
            if replaced:
                modifications[kind] += 1
        if _HTML_ENTITY.search(cleaned):
            cleaned = html.unescape(cleaned)
            modifications["html_entities_unescaped"] += 1

        tidy = _WHITESPACE_RUN.sub(" ", _INVISIBLE.sub("", cleaned)).strip()
        if tidy != cleaned:
            modifications["whitespace_normalized"] += 1
        return tidy


class DuplicateFeedbackFilter:
    """Keeps the first of several records with the same author and the same text.

    Matching is case- and whitespace-insensitive. Identical short texts from *different*
    authors ("Great product") are distinct opinions and are kept. When a source has no author
    field, records about the same entity with the same text are treated as duplicates.
    """

    name = "duplicate_filter"

    def __init__(self, *, author_metadata_key: str | None = "user_id") -> None:
        self._author_key = author_metadata_key

    def apply(self, items: Sequence[FeedbackItem]) -> StepOutcome:
        outcome = StepOutcome(items=[])
        seen: set[tuple[str, str, str]] = set()
        for item in items:
            key = (item.source, self._author_of(item), item.text.casefold())
            if key in seen:
                outcome.exclusions[DUPLICATE_TEXT] += 1
                continue
            seen.add(key)
            outcome.items.append(item)
        return outcome

    def _author_of(self, item: FeedbackItem) -> str:
        author = item.metadata.get(self._author_key) if self._author_key else None
        return f"author:{author}" if author else f"entity:{item.entity_id}"


def _with_cleaned_values(item: FeedbackItem, *, text: str, title: str | None) -> FeedbackItem:
    if text == item.text and title == item.title:
        return item
    metadata = dict(item.metadata)
    if text != item.text:
        metadata.setdefault("original_text", item.text)
    if title != item.title and item.title is not None:
        metadata.setdefault("original_title", item.title)
    return item.model_copy(update={"text": text, "title": title, "metadata": metadata})
