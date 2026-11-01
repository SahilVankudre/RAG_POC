"""LLM adapter for any API that speaks the OpenAI chat completions format.

OpenAI, DeepSeek and local Ollama differ only in base URL, key and model name, so switching
between them is a configuration change. The README lists working presets.
"""

import logging
import re
import time
from typing import Any

import httpx2
import openai
from openai import OpenAI

from rag_poc.lib import LLMError, LLMResponse

logger = logging.getLogger(__name__)

# The SDK refuses an empty key, but local servers such as Ollama ignore whatever is sent.
_PLACEHOLDER_API_KEY = "not-needed"
# Some local models put their reasoning in the reply itself rather than in a separate field.
_REASONING_BLOCK = re.compile(r"<think>.*?(</think>|\Z)", re.DOTALL)


class LLMService:
    """Sends prompts to the configured model and returns usable text or a clear ``LLMError``."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str | None,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: float,
        max_retries: int,
        extra_body: dict[str, Any] | None = None,
        http_client: httpx2.Client | None = None,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._timeout_seconds = timeout_seconds
        self._extra_body = extra_body or None
        # Both are always explicit: the SDK would otherwise fall back to OPENAI_API_KEY and
        # OPENAI_BASE_URL, which could send one provider's key to another provider.
        self._client = OpenAI(
            api_key=api_key or _PLACEHOLDER_API_KEY,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
            http_client=http_client,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> LLMResponse:
        if not prompt.strip():
            raise LLMError("The prompt must not be empty.")
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        started = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=self._temperature,
                # max_tokens is the limit field shared by every OpenAI-format server we target.
                max_tokens=self._max_output_tokens,
                extra_body=self._extra_body,
            )
        except openai.OpenAIError as error:
            raise self._describe_failure(error) from error
        return self._to_response(completion, seconds=time.perf_counter() - started)

    def _to_response(self, completion: Any, *, seconds: float) -> LLMResponse:
        choices = getattr(completion, "choices", None)
        if not choices:
            # Gateways such as OpenRouter can report a provider failure as status 200.
            if failure := getattr(completion, "error", None):
                message = failure.get("message") if isinstance(failure, dict) else failure
                raise LLMError(f"{self._model_name} failed: {message}")
            # A wrong base URL can return a web page with status 200, passed through by the SDK.
            raise LLMError(
                f"{self._base_url} did not return a chat completion. Check RAG_LLM_BASE_URL."
            )
        choice = choices[0]
        finish_reason = choice.finish_reason or "stop"
        text = _REASONING_BLOCK.sub("", choice.message.content or "").strip()
        if not text:
            raise LLMError(self._describe_empty_reply(finish_reason))

        response = LLMResponse(
            text=text,
            model_name=completion.model or self._model_name,
            finish_reason=finish_reason,
            input_tokens=completion.usage.prompt_tokens if completion.usage else None,
            output_tokens=completion.usage.completion_tokens if completion.usage else None,
            seconds=seconds,
        )
        if not response.is_complete:
            logger.warning("LLM reply stopped early (finish reason: %s)", finish_reason)
        return response

    def _describe_empty_reply(self, finish_reason: str) -> str:
        if finish_reason == "length":
            return (
                f"{self._model_name} reached the output limit of {self._max_output_tokens} tokens "
                "before writing an answer, which usually means it spent them on reasoning. Raise "
                "RAG_LLM_MAX_OUTPUT_TOKENS or turn its reasoning off with RAG_LLM_EXTRA_BODY."
            )
        return f"{self._model_name} returned no text (finish reason: {finish_reason})."

    def _describe_failure(self, error: openai.OpenAIError) -> LLMError:
        name = self._model_name
        if isinstance(error, openai.APITimeoutError):
            return LLMError(
                f"{name} did not respond within {self._timeout_seconds:g}s. Raise "
                "RAG_LLM_TIMEOUT_SECONDS or use a faster model."
            )
        if isinstance(error, openai.APIConnectionError):
            return LLMError(
                f"Could not connect to {self._base_url}. Check RAG_LLM_BASE_URL and, for a local "
                "model, that the server is running."
            )
        if isinstance(error, openai.AuthenticationError):
            return LLMError(f"{self._base_url} rejected the API key. Check RAG_LLM_API_KEY.")
        if isinstance(error, openai.NotFoundError):
            return LLMError(
                f"{self._base_url} has no model {name!r}, or the URL is wrong. Check "
                "RAG_LLM_MODEL_NAME and RAG_LLM_BASE_URL."
            )
        if isinstance(error, openai.RateLimitError):
            return LLMError(f"{name} is rate limited or out of quota: {error.message}")
        if isinstance(error, openai.APIStatusError):
            return LLMError(f"{name} failed with status {error.status_code}: {error.message}")
        return LLMError(f"The LLM request failed: {error}")
