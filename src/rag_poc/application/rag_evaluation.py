"""Use case: run the question set through the RAG pipeline and keep every result for review."""

import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

from rag_poc.application.rag_service import RAGService
from rag_poc.evaluation import EvaluationStore
from rag_poc.lib import EvaluationQuestion, EvaluationRecord, EvaluationRun, LLMError

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


class RAGEvaluation:
    """Automates what can be checked mechanically and records what a person must judge.

    The run confirms whether each question was answered or refused as expected. Whether
    retrieved reviews are relevant and cited reviews support the claims is judged by hand from
    the saved results, so a failed check here and a clean run both need reading.
    """

    def __init__(self, *, rag: RAGService, store: EvaluationStore, model_name: str) -> None:
        self._rag = rag
        self._store = store
        self._model_name = model_name

    def run(self, on_progress: ProgressCallback | None = None) -> EvaluationRun:
        questions = self._store.load_questions()
        records = []
        for done, question in enumerate(questions, start=1):
            records.append(self._evaluate(question))
            if on_progress:
                on_progress(done, len(questions))

        run = EvaluationRun(
            records=tuple(records),
            model_name=self._model_name,
            created_at=datetime.now(UTC),
        )
        return replace(run, location=self._store.save_run(run))

    def _evaluate(self, question: EvaluationQuestion) -> EvaluationRecord:
        """One failed call shouldn't discard the answers already paid for."""
        try:
            response = self._rag.ask(question.question, filters=question.filters)
        except LLMError as error:
            logger.warning("%s failed: %s", question.question_id, error)
            return EvaluationRecord(question=question, response=None, error=str(error))
        return EvaluationRecord(question=question, response=response)
