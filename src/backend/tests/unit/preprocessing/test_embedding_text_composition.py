import pytest

from rag_poc.lib import FeedbackItem
from rag_poc.preprocessing import compose_embedding_text


@pytest.mark.parametrize(
    ("title", "text", "expected", "combined"),
    [
        ("Broke fast", "The hinge snapped.", "Broke fast. The hinge snapped.", True),
        ("Love it!", "Smells amazing.", "Love it! Smells amazing.", True),
        ("Five Stars", "Smells amazing.", "Smells amazing.", False),
        ("one star.", "Rash after one use.", "Rash after one use.", False),
        ("Great value", "Great value for money.", "Great value for money.", False),
        (None, "No title here.", "No title here.", False),
    ],
)
def test_title_is_combined_only_when_it_adds_information(
    title: str | None, text: str, expected: str, combined: bool
) -> None:
    item = FeedbackItem(feedback_id="1", source="amazon", entity_id="B1", text=text, title=title)

    assert compose_embedding_text(item) == (expected, combined)
