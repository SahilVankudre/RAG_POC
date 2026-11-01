"""The outcome of one LLM generation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Generated text plus what callers need to judge and report on it."""

    text: str
    model_name: str
    finish_reason: str
    input_tokens: int | None
    output_tokens: int | None
    seconds: float

    @property
    def is_complete(self) -> bool:
        """False when generation stopped early, e.g. at the output limit."""
        return self.finish_reason == "stop"
