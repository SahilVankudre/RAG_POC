import json
from datetime import UTC, datetime
from pathlib import Path

from rag_poc.lib import DataQualityReport, FeedbackItem, PreparedFeedback
from rag_poc.preprocessing.parquet_processed_feedback_store import ParquetProcessedFeedbackStore


def test_prepared_feedback_and_report_survive_a_round_trip(tmp_path: Path) -> None:
    prepared = [
        PreparedFeedback(
            item=FeedbackItem(
                feedback_id="amz_1",
                source="amazon",
                entity_id="B1",
                text="Broke in a week 😞",
                title="Snapped",
                rating=1.0,
                timestamp=datetime(2021, 3, 4, 5, 6, 7, tzinfo=UTC),
                metadata={"user_id": "U1", "verified_purchase": True, "helpful_vote": 2},
            ),
            embedding_text="Snapped. Broke in a week 😞",
        ),
        PreparedFeedback(
            item=FeedbackItem(feedback_id="amz_2", source="amazon", entity_id="B2", text="Ok"),
            embedding_text="Ok",
        ),
    ]
    report = DataQualityReport(
        source_name="amazon:test",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        input_records=3,
        prepared_records=2,
        adapter_rejections={"missing_text": 1},
        cleaning_exclusions={},
        text_modifications={},
        titles_combined=1,
        short_text_records=1,
        long_text_records=0,
    )
    store = ParquetProcessedFeedbackStore(
        location=tmp_path / "items.parquet", report_location=tmp_path / "report.json"
    )

    store.save(prepared, report)

    assert store.load() == prepared
    saved_report = json.loads(store.report_location.read_text(encoding="utf-8"))
    assert saved_report["is_reconciled"] is True
    assert saved_report["excluded_records"] == 1
