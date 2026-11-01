"""LLM module: generates text from a prompt, independently of retrieval.

Exposes the ``TextGenerator`` port. ``LLMService`` is the adapter for any API that speaks the
OpenAI chat completions format; the composition root imports it lazily.
"""

from rag_poc.llm.llm_ports import TextGenerator

__all__ = ["TextGenerator"]
