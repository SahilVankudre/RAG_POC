"""A cleaned feedback item paired with the exact text that will be embedded."""

from dataclasses import dataclass

from rag_poc.lib.feedback_item import FeedbackItem


@dataclass(frozen=True, slots=True)
class PreparedFeedback:
    """Output of preprocessing and input to embedding.

    ``item`` holds the cleaned text and title (originals are kept in ``item.metadata`` under
    ``original_text`` / ``original_title`` whenever cleaning changed them). ``embedding_text``
    is what the model encodes, e.g. a useful title combined with the body.
    """

    item: FeedbackItem
    embedding_text: str

    @property
    def feedback_id(self) -> str:
        return self.item.feedback_id
