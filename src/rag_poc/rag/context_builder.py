"""Builds the block of reviews the LLM reads, keeping every review traceable to its id."""

from collections.abc import Sequence
from dataclasses import dataclass

from rag_poc.lib import RAGSource, SearchResult

_TRUNCATION_MARK = " [truncated]"


@dataclass(frozen=True, slots=True)
class ReviewContext:
    """Formatted reviews plus the sources their labels refer to."""

    text: str
    sources: tuple[RAGSource, ...]


def build_review_context(
    reviews: Sequence[SearchResult], *, review_max_chars: int
) -> ReviewContext:
    """Label reviews ``R1``, ``R2``, ... in rank order and format them for the LLM.

    Models cite a short label reliably, whereas a 16-character hash is easy to mangle, so the
    real id sits beside the label and citations are mapped back through ``sources``.
    """
    sources = tuple(
        RAGSource(label=f"R{position}", review=review)
        for position, review in enumerate(reviews, start=1)
    )
    blocks = (_format_review(source, review_max_chars) for source in sources)
    return ReviewContext(text="\n\n".join(blocks), sources=sources)


def _format_review(source: RAGSource, max_chars: int) -> str:
    review = source.review
    rating = "none" if review.rating is None else f"{review.rating:g}/5"
    date = f' date="{review.timestamp:%Y-%m-%d}"' if review.timestamp else ""
    lines = [
        f'<review label="{source.label}" feedback_id="{review.feedback_id}" '
        f'product="{review.entity_id}" rating="{rating}"{date}>'
    ]
    if review.title:
        lines.append(f"Title: {_as_review_data(review.title, max_chars)}")
    lines.append(_as_review_data(review.text, max_chars))
    lines.append("</review>")
    return "\n".join(lines)


def _as_review_data(text: str, max_chars: int) -> str:
    """Flatten and shorten review text so it can't close its own tag or crowd out other reviews."""
    flat = " ".join(text.split())
    if len(flat) > max_chars:
        flat = flat[:max_chars].rstrip() + _TRUNCATION_MARK
    return flat.replace("<", "&lt;").replace(">", "&gt;")
