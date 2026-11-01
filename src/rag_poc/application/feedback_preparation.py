"""Use case: turn the raw review sample into clean, embedding-ready feedback (spec §4–6)."""

import logging
from datetime import UTC, datetime

from rag_poc.ingestion import FeedbackAdapter, ReviewSampleStore
from rag_poc.lib import DataQualityReport, DatasetError
from rag_poc.preprocessing import FeedbackCleaningPipeline, ProcessedFeedbackStore

logger = logging.getLogger(__name__)


class FeedbackPreparation:
    """Sample -> source adapter -> cleaning pipeline -> processed store, with full accounting."""

    def __init__(
        self,
        *,
        sample_store: ReviewSampleStore,
        adapter: FeedbackAdapter,
        pipeline: FeedbackCleaningPipeline,
        processed_store: ProcessedFeedbackStore,
    ) -> None:
        self._sample_store = sample_store
        self._adapter = adapter
        self._pipeline = pipeline
        self._processed_store = processed_store

    def prepare(self) -> DataQualityReport:
        if not self._sample_store.exists():
            raise DatasetError(
                f"No review sample at {self._sample_store.location}. Run `rag-poc sample` first."
            )
        records = self._sample_store.load()
        adaptation = self._adapter.adapt(records)
        outcome = self._pipeline.run(adaptation.items)

        report = DataQualityReport(
            source_name=f"{self._adapter.source}:{self._sample_store.location.name}",
            created_at=datetime.now(UTC),
            input_records=len(records),
            prepared_records=len(outcome.prepared),
            adapter_rejections=adaptation.rejection_counts,
            cleaning_exclusions=outcome.exclusions,
            text_modifications=outcome.modifications,
            titles_combined=outcome.titles_combined,
            short_text_records=outcome.short_text_records,
            long_text_records=outcome.long_text_records,
        )
        if not report.is_reconciled:  # a bug, not a data problem: fail loudly
            raise RuntimeError(f"Data-quality report does not reconcile: {report.to_dict()}")

        self._processed_store.save(outcome.prepared, report)
        logger.info(
            "Prepared %s feedback items -> %s",
            f"{report.prepared_records:,}",
            self._processed_store.location,
        )
        return report
