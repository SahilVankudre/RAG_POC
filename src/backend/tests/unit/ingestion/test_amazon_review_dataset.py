import json
from pathlib import Path

from rag_poc.ingestion.amazon import read_amazon_review_file


def test_review_file_is_parsed_with_unneeded_fields_skipped(tmp_path: Path) -> None:
    records = [
        {
            "rating": 5.0,
            "title": "Love it",
            "text": "Smells great",
            "images": [{"large_image_url": "https://example.com/a.jpg"}],
            "asin": "B000000001",
            "parent_asin": "B000000001",
            "user_id": "U1",
            "timestamp": 1588687728923,
            "helpful_vote": 2,
            "verified_purchase": True,
        },
        {"rating": 1.0, "text": "Broke in a week", "asin": "B000000002", "user_id": "U2"},
    ]
    review_file = tmp_path / "All_Beauty.jsonl"
    review_file.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    reviews = read_amazon_review_file(review_file)

    assert len(reviews) == 2
    assert "images" not in reviews.columns
    assert reviews.loc[0, "timestamp"] == 1588687728923
    assert bool(reviews.loc[0, "verified_purchase"]) is True
    assert reviews[["title", "timestamp", "verified_purchase"]].iloc[1].isna().all()
