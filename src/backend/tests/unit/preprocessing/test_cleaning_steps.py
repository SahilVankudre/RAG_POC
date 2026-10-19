from rag_poc.lib import FeedbackItem
from rag_poc.preprocessing import (
    BLANK_AFTER_CLEANING,
    DUPLICATE_TEXT,
    DuplicateFeedbackFilter,
    ReviewTextNormalization,
)


def _item(feedback_id: str, text: str, **fields: object) -> FeedbackItem:
    return FeedbackItem(
        feedback_id=feedback_id, source="amazon", entity_id="B1", text=text, **fields
    )


def test_markup_and_whitespace_are_cleaned_and_originals_preserved() -> None:
    raw = "Works&#34;great&#34;<br /><br />[[VIDEOID:abc123]]  but   the cap​ broke."
    [item] = ReviewTextNormalization().apply([_item("1", raw, title="  Good  ")]).items

    assert item.text == 'Works"great" but the cap broke.'
    assert item.title == "Good"
    assert item.metadata == {"original_text": raw, "original_title": "  Good  "}


def test_video_only_reviews_fall_back_to_an_informative_title() -> None:
    outcome = ReviewTextNormalization().apply(
        [
            _item("1", "[[VIDEOID:abc]]", title="Colors were wrong"),
            _item("2", "[[VIDEOID:def]] <br/>", title="Five Stars"),
            _item("3", "Already clean"),
        ]
    )

    assert [item.text for item in outcome.items] == ["Colors were wrong", "Already clean"]
    assert outcome.exclusions == {BLANK_AFTER_CLEANING: 1}
    assert outcome.modifications["title_used_as_text"] == 1
    assert outcome.items[1].metadata == {}


def test_duplicates_are_matched_per_author_not_across_authors() -> None:
    outcome = DuplicateFeedbackFilter(author_metadata_key="user_id").apply(
        [
            _item("1", "Great product", metadata={"user_id": "U1"}),
            _item("2", "great PRODUCT", metadata={"user_id": "U1"}),  # same author, repost
            _item("3", "Great product", metadata={"user_id": "U2"}),  # different author
            _item("4", "No author"),
            _item("5", "no author"),  # same entity, no author -> duplicate
        ]
    )

    assert [item.feedback_id for item in outcome.items] == ["1", "3", "4"]
    assert outcome.exclusions == {DUPLICATE_TEXT: 2}
