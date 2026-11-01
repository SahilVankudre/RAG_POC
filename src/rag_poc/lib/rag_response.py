"""The structured outcome of answering a question from retrieved feedback."""

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rag_poc.lib.llm_response import LLMResponse
from rag_poc.lib.search_index_types import SearchResult


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class InsufficiencyReason(StrEnum):
    """Why a question was not answered, from cheapest check to latest."""

    NO_MATCHING_REVIEWS = "no_matching_reviews"
    LOW_RELEVANCE = "low_relevance"
    MODEL_DECLINED = "model_declined"
    NO_VALID_CITATIONS = "no_valid_citations"


@dataclass(frozen=True, slots=True)
class RAGSource:
    """A retrieved review and the short label the LLM cites it by, e.g. ``R3``."""

    label: str
    review: SearchResult

    def to_dict(self) -> dict[str, Any]:
        review = asdict(self.review)
        timestamp = self.review.timestamp
        return {"label": self.label} | review | {"timestamp": timestamp and timestamp.isoformat()}


@dataclass(frozen=True, slots=True)
class RAGResponse:
    """An answer plus everything needed to trace it back to the reviews behind it.

    ``retrieved`` holds every review fetched and ``cited`` only those the answer relies on.
    ``invalid_citations`` lists labels the model made up, which are removed from ``answer``.
    A citation being valid means the review exists, not that it supports the claim.
    """

    question: str
    status: AnswerStatus
    reason: InsufficiencyReason | None
    answer: str
    cited: tuple[RAGSource, ...]
    retrieved: tuple[RAGSource, ...]
    invalid_citations: tuple[str, ...]
    relevance: float | None  # how far the feedback overall discusses the topic; None if unmeasured
    context: str
    llm: LLMResponse | None  # None when no LLM call was made

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, for APIs and evaluation records."""
        return {
            "question": self.question,
            "status": self.status.value,
            "reason": self.reason.value if self.reason else None,
            "answer": self.answer,
            "cited": [source.to_dict() for source in self.cited],
            "retrieved": [source.to_dict() for source in self.retrieved],
            "invalid_citations": list(self.invalid_citations),
            "relevance": self.relevance,
            "context": self.context,
            "llm": self.llm and asdict(self.llm) | {"is_complete": self.llm.is_complete},
        }
