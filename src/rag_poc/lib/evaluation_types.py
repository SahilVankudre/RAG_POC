"""Types for running a fixed question set through the RAG pipeline."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_poc.lib.rag_response import AnswerStatus, RAGResponse
from rag_poc.lib.search_index_types import SearchFilters


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    question_id: str
    category: str
    question: str
    expected_status: AnswerStatus
    what_to_verify: str
    filters: SearchFilters | None = None


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One question's outcome; ``response`` is None if the question failed with ``error``."""

    question: EvaluationQuestion
    response: RAGResponse | None
    error: str | None = None

    @property
    def outcome_as_expected(self) -> bool:
        return self.response is not None and (self.response.status == self.question.expected_status)

    def to_dict(self) -> dict[str, Any]:
        filters = self.question.filters
        return {
            "id": self.question.question_id,
            "category": self.question.category,
            "filters": filters.describe() if filters else "default",
            "expected": self.question.expected_status.value,
            "outcome_as_expected": self.outcome_as_expected,
            "what_to_verify": self.question.what_to_verify,
            "error": self.error,
            "response": self.response and self.response.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """Automatic results of a run. Whether answers are true is judged by a person."""

    records: tuple[EvaluationRecord, ...]
    model_name: str
    created_at: datetime
    location: Path | None = None

    @property
    def as_expected(self) -> int:
        return sum(record.outcome_as_expected for record in self.records)

    @property
    def failed(self) -> int:
        return sum(record.error is not None for record in self.records)

    @property
    def invalid_citations(self) -> int:
        return sum(
            len(record.response.invalid_citations) for record in self.records if record.response
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "created_at": self.created_at.isoformat(),
            "questions": len(self.records),
            "as_expected": self.as_expected,
            "failed": self.failed,
            "invalid_citations": self.invalid_citations,
            "records": [record.to_dict() for record in self.records],
        }
