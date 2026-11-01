"""RAG module: turns retrieved reviews and a question into a prompt for the LLM.

Holds no retrieval or LLM logic of its own; ``application.rag_service`` connects all three.
"""

from rag_poc.rag.context_builder import ReviewContext, build_review_context
from rag_poc.rag.prompts import INSUFFICIENT_EVIDENCE_MARKER, RAG_SYSTEM_PROMPT, build_rag_prompt
from rag_poc.rag.reply_interpretation import InterpretedReply, interpret_llm_reply

__all__ = [
    "INSUFFICIENT_EVIDENCE_MARKER",
    "RAG_SYSTEM_PROMPT",
    "InterpretedReply",
    "ReviewContext",
    "build_rag_prompt",
    "build_review_context",
    "interpret_llm_reply",
]
