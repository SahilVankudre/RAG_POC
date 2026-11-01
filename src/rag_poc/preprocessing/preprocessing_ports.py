"""Ports of the preprocessing module."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from rag_poc.lib import DataQualityReport, PreparedFeedback


class ProcessedFeedbackStore(Protocol):
    """Persists embedding-ready feedback and its data-quality report."""

    @property
    def location(self) -> Path: ...

    @property
    def report_location(self) -> Path: ...

    def exists(self) -> bool: ...

    def save(self, prepared: Sequence[PreparedFeedback], report: DataQualityReport) -> None: ...

    def load(self) -> list[PreparedFeedback]: ...
