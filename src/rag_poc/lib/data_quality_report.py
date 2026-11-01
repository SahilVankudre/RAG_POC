"""Accounting of every record from the raw sample to embedding-ready feedback (spec §6)."""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """How many records entered preparation, how many came out, and why the rest were dropped.

    ``input_records`` always equals ``prepared_records`` plus every adapter rejection and
    cleaning exclusion; ``is_reconciled`` checks that nothing went missing unaccounted.
    """

    source_name: str
    created_at: datetime
    input_records: int
    prepared_records: int
    adapter_rejections: dict[str, int]
    cleaning_exclusions: dict[str, int]
    text_modifications: dict[str, int]
    titles_combined: int
    short_text_records: int
    long_text_records: int

    @property
    def excluded_records(self) -> int:
        return sum(self.adapter_rejections.values()) + sum(self.cleaning_exclusions.values())

    @property
    def is_reconciled(self) -> bool:
        return self.input_records == self.prepared_records + self.excluded_records

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation, including the derived totals."""
        return asdict(self) | {
            "created_at": self.created_at.isoformat(),
            "excluded_records": self.excluded_records,
            "is_reconciled": self.is_reconciled,
        }
