from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, Literal

from long_document_indexing.domain.runs import UsageRecord
from long_document_indexing.models.base import GenerationRequest, GenerationResponse

AuthMode = Literal["api_key", "azure_default_credential"]
ResponseFormat = Literal["json_object", "text"]


class OpenAICompatibleTextGenerationClient:
    """Text generation client for Foundry/Azure OpenAI compatible v1 endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        deployment: str,
        api_key: str | None = None,
        api_key_env: str = "AZURE_INFERENCE_CREDENTIAL",
        auth_mode: AuthMode = "api_key",
        azure_scope: str = "https://ai.azure.com/.default",
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        timeout_seconds: float = 60.0,
        response_format: ResponseFormat = "json_object",
        client: Any | None = None,
        token_provider: Callable[[], str] | None = None,
    ) -> None:
        self.base_url = base_url
        self.deployment = deployment
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.auth_mode = auth_mode
        self.azure_scope = azure_scope
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.response_format = response_format
        self._client = client
        self._token_provider = token_provider

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        return await asyncio.to_thread(self._generate_sync, request)

    def _generate_sync(self, request: GenerationRequest) -> GenerationResponse:
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": self.temperature,
        }
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens
        if self.response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        completion = self._openai_client().chat.completions.create(**kwargs)
        choice = _first_choice(completion)
        duration_ms = (time.perf_counter() - started) * 1000.0
        usage = _usage_record(completion, duration_ms=duration_ms)

        return GenerationResponse(
            content=_message_content(choice),
            usage=usage,
            metadata={
                "client": self.__class__.__name__,
                "provider": "openai_compatible",
                "deployment": self.deployment,
                "model": _get(completion, "model"),
                "response_id": _get(completion, "id"),
                "finish_reason": _get(choice, "finish_reason"),
                "prompt_name": request.prompt_name,
            },
        )

    def _openai_client(self) -> Any:
        if self._client is None:
            self._client = self._build_openai_client()
        return self._client

    def _build_openai_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The real Foundry/OpenAI-compatible generator requires the optional "
                "`foundry` extra. Install it with `uv sync --extra foundry`."
            ) from exc

        kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "timeout": self.timeout_seconds,
        }
        if self.auth_mode == "api_key":
            kwargs["api_key"] = self._required_api_key()
        elif self.auth_mode == "azure_default_credential":
            kwargs["api_key"] = self._required_token_provider()
        else:
            raise ValueError(f"unsupported auth mode: {self.auth_mode}")
        return OpenAI(**kwargs)

    def _required_api_key(self) -> str:
        if self.api_key:
            return self.api_key

        import os

        value = os.environ.get(self.api_key_env)
        if value:
            return value
        raise RuntimeError(
            f"missing API key for real generator; set environment variable {self.api_key_env}"
        )

    def _required_token_provider(self) -> Callable[[], str]:
        if self._token_provider is not None:
            return self._token_provider
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except ImportError as exc:
            raise RuntimeError(
                "Azure default credential auth requires the optional `foundry` extra. "
                "Install it with `uv sync --extra foundry`."
            ) from exc

        self._token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            self.azure_scope,
        )
        return self._token_provider


def _first_choice(completion: Any) -> Any:
    choices = _get(completion, "choices", [])
    if not choices:
        raise RuntimeError("model response did not include any choices")
    return choices[0]


def _message_content(choice: Any) -> str:
    message = _get(choice, "message")
    content = _get(message, "content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        text_parts = [
            str(_get(part, "text", ""))
            for part in content
            if _get(part, "type") in {None, "text", "output_text"}
        ]
        joined = "".join(text_parts).strip()
        if joined:
            return joined
    raise RuntimeError("model response choice did not include text content")


def _usage_record(completion: Any, *, duration_ms: float) -> UsageRecord:
    usage = _get(completion, "usage")
    return UsageRecord(
        input_tokens=_int_field(usage, "prompt_tokens", "input_tokens"),
        output_tokens=_int_field(usage, "completion_tokens", "output_tokens"),
        model_calls=1,
        duration_ms=duration_ms,
    )


def _int_field(value: Any, *names: str) -> int:
    for name in names:
        raw = _get(value, name)
        if raw is not None:
            return int(raw)
    return 0


def _get(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
