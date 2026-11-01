"""Use case: answer a question from customer feedback, keeping the sources behind the answer."""

from statistics import mean

from rag_poc.lib import AnswerStatus, InsufficiencyReason, RAGResponse, RAGSource, SearchFilters
from rag_poc.llm import TextGenerator
from rag_poc.rag import (
    RAG_SYSTEM_PROMPT,
    build_rag_prompt,
    build_review_context,
    interpret_llm_reply,
)
from rag_poc.retrieval import FeedbackRetriever

_NO_REVIEWS_ANSWER = "No customer reviews matched this question, so it cannot be answered."
_LOW_RELEVANCE_ANSWER = (
    "The customer reviews do not appear to discuss this topic, so there is not enough evidence "
    "to answer."
)

# Relevance is always the mean of this many unfiltered matches, whatever the request's top_k and
# filters, so the threshold keeps one meaning: does the feedback discuss this topic at all? A
# larger top_k pulls a mean down with weaker matches, and filters shrink the pool: on-topic
# questions about a single product scored 0.18-0.22, well below the threshold.
_RELEVANCE_WINDOW = 10


class RAGService:
    """Answers from retrieved reviews, and says so when they can't support an answer.

    Two checks guard against unsupported answers. A relevance check rejects off-topic questions
    without calling the LLM. It can't judge answerability, because a review can match a
    question's topic without answering it, so the LLM must also be able to decline.
    """

    def __init__(
        self,
        *,
        retriever: FeedbackRetriever,
        llm: TextGenerator,
        default_top_k: int,
        review_max_chars: int,
        min_relevance: float,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._default_top_k = default_top_k
        self._review_max_chars = review_max_chars
        self._min_relevance = min_relevance

    def ask(
        self, question: str, top_k: int | None = None, filters: SearchFilters | None = None
    ) -> RAGResponse:
        question = question.strip()
        reviews = self._retriever.search(
            question, top_k=top_k or self._default_top_k, filters=filters
        )
        context = build_review_context(reviews, review_max_chars=self._review_max_chars)
        if not context.sources:
            return _unanswered(
                question, InsufficiencyReason.NO_MATCHING_REVIEWS, _NO_REVIEWS_ANSWER
            )

        relevance = self._topic_relevance(question)
        if relevance < self._min_relevance:
            return _unanswered(
                question,
                InsufficiencyReason.LOW_RELEVANCE,
                _LOW_RELEVANCE_ANSWER,
                retrieved=context.sources,
                relevance=relevance,
            )

        generated = self._llm.generate(
            build_rag_prompt(question, context.text), system_prompt=RAG_SYSTEM_PROMPT
        )
        reply = interpret_llm_reply(generated.text, context.sources)
        return RAGResponse(
            question=question,
            status=reply.status,
            reason=reply.reason,
            answer=reply.answer,
            cited=reply.cited,
            retrieved=context.sources,
            invalid_citations=reply.invalid_citations,
            relevance=relevance,
            context=context.text,
            llm=generated,
        )

    def _topic_relevance(self, question: str) -> float:
        matches = self._retriever.search(question, top_k=_RELEVANCE_WINDOW)
        return mean(match.score for match in matches) if matches else 0.0


def _unanswered(
    question: str,
    reason: InsufficiencyReason,
    answer: str,
    *,
    retrieved: tuple[RAGSource, ...] = (),
    relevance: float | None = None,
) -> RAGResponse:
    """A response for a question stopped before reaching the LLM."""
    return RAGResponse(
        question=question,
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        reason=reason,
        answer=answer,
        cited=(),
        retrieved=retrieved,
        invalid_citations=(),
        relevance=relevance,
        context="",
        llm=None,
    )
