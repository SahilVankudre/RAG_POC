"""Ports of the evaluation module."""

from pathlib import Path
from typing import Protocol

from rag_poc.lib import EvaluationQuestion, EvaluationRun


class EvaluationStore(Protocol):
    """Supplies the question set and persists run results for manual review."""

    def load_questions(self) -> list[EvaluationQuestion]: ...

    def save_run(self, run: EvaluationRun) -> Path:
        """Save the run and return where it went."""
        ...
