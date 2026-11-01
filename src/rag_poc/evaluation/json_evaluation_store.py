"""Reads the question set from JSON and writes each run to a timestamped JSON file."""

import json
from pathlib import Path
from typing import Any

from rag_poc.lib import (
    AnswerStatus,
    EvaluationError,
    EvaluationQuestion,
    EvaluationRun,
    SearchFilters,
)


class JsonEvaluationStore:
    def __init__(self, *, questions_location: Path, runs_location: Path) -> None:
        self._questions_location = questions_location
        self._runs_location = runs_location

    def load_questions(self) -> list[EvaluationQuestion]:
        try:
            entries = json.loads(self._questions_location.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise EvaluationError(f"No question set at {self._questions_location}.") from error
        except json.JSONDecodeError as error:
            raise EvaluationError(
                f"{self._questions_location} is not valid JSON: {error}"
            ) from error
        if not isinstance(entries, list) or not entries:
            raise EvaluationError(f"{self._questions_location} must be a non-empty JSON list.")

        questions = [self._to_question(entry) for entry in entries]
        ids = [question.question_id for question in questions]
        if len(set(ids)) != len(ids):
            raise EvaluationError(f"{self._questions_location} repeats a question id.")
        return questions

    def save_run(self, run: EvaluationRun) -> Path:
        self._runs_location.mkdir(parents=True, exist_ok=True)
        location = self._runs_location / f"run_{run.created_at:%Y%m%d_%H%M%S}.json"
        location.write_text(
            json.dumps(run.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return location

    def _to_question(self, entry: dict[str, Any]) -> EvaluationQuestion:
        try:
            filters = entry.get("filters")
            return EvaluationQuestion(
                question_id=entry["id"],
                category=entry["category"],
                question=entry["question"],
                expected_status=AnswerStatus(entry["expected"]),
                what_to_verify=entry["what_to_verify"],
                filters=SearchFilters(**filters) if filters else None,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationError(
                f"Invalid question in {self._questions_location}: {entry!r} ({error!r})"
            ) from error
