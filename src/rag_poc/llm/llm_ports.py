"""Ports of the LLM module."""

from typing import Protocol

from rag_poc.lib import LLMResponse


class TextGenerator(Protocol):
    """Turns a prompt into generated text; callers never see which provider answered."""

    @property
    def model_name(self) -> str: ...

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> LLMResponse:
        """Generate a response, raising ``LLMError`` if no usable text comes back.

        ``system_prompt`` carries standing instructions, kept apart from ``prompt`` so that
        text quoted in the prompt is less able to pass itself off as instructions.
        """
        ...
