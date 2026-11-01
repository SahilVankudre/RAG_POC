"""Turns the LLM's raw reply into a checked result: declined, or answered with valid citations."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag_poc.lib import AnswerStatus, InsufficiencyReason, RAGSource
from rag_poc.rag.prompts import INSUFFICIENT_EVIDENCE_MARKER

UNCITED_REPLY_ANSWER = (
    "The model's reply cited none of the retrieved reviews, so it is not shown as an answer."
)
_DECLINED_ANSWER = "The available customer feedback does not provide sufficient evidence."

# Models sometimes dress the marker up as **INSUFFICIENT_EVIDENCE** or add punctuation after it.
_DECLINED_REPLY = re.compile(rf"[\s*_#`]*{INSUFFICIENT_EVIDENCE_MARKER}[\s*_`:.\-]*", re.IGNORECASE)
# Matches [R1] and [R1, R2], with any spaces in front so a removed citation leaves no gap.
_CITATION = re.compile(r"( *)\[\s*(R\d+(?:\s*,\s*R\d+)*)\s*\]")


@dataclass(frozen=True, slots=True)
class InterpretedReply:
    status: AnswerStatus
    reason: InsufficiencyReason | None
    answer: str
    cited: tuple[RAGSource, ...]
    invalid_citations: tuple[str, ...]


def interpret_llm_reply(reply: str, sources: Sequence[RAGSource]) -> InterpretedReply:
    """Decide what the reply amounts to, trusting only citations of reviews that were retrieved."""
    by_label = {source.label: source for source in sources}
    cited: dict[str, RAGSource] = {}
    invalid: dict[str, None] = {}

    def keep_valid_labels(match: re.Match[str]) -> str:
        labels = [label.strip() for label in match[2].split(",")]
        kept = [label for label in labels if label in by_label]
        cited.update((label, by_label[label]) for label in kept)
        invalid.update((label, None) for label in labels if label not in by_label)
        return match[1] + "".join(f"[{label}]" for label in kept) if kept else ""

    answer = _CITATION.sub(keep_valid_labels, reply).strip()

    if declined := _DECLINED_REPLY.match(answer):
        return InterpretedReply(
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            reason=InsufficiencyReason.MODEL_DECLINED,
            answer=answer[declined.end() :].strip() or _DECLINED_ANSWER,
            cited=(),
            invalid_citations=tuple(invalid),
        )
    if not cited:
        return InterpretedReply(
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            reason=InsufficiencyReason.NO_VALID_CITATIONS,
            answer=UNCITED_REPLY_ANSWER,
            cited=(),
            invalid_citations=tuple(invalid),
        )
    return InterpretedReply(
        status=AnswerStatus.ANSWERED,
        reason=None,
        answer=answer,
        cited=tuple(cited.values()),
        invalid_citations=tuple(invalid),
    )
