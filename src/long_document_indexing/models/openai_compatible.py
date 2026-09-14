from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel

from long_document_indexing.domain.runs import UsageRecord
from long_document_indexing.models.base import GenerationRequest, GenerationResponse

AuthMode = Literal["api_key", "azure_default_credential"]
GeneratorApi = Literal["chat_completions", "responses"]
ResponseFormat = Literal["structured", "json_object", "text"]


class OpenAICompatibleTextGenerationClient:
    """Text generation client for Foundry/Azure OpenAI compatible v1 endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        deployment: str,
        api: GeneratorApi = "chat_completions",
        api_key: str | None = None,
        api_key_env: str = "AZURE_INFERENCE_CREDENTIAL",
        auth_mode: AuthMode = "api_key",
        azure_scope: str = "https://ai.azure.com/.default",
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 12,
        response_format: ResponseFormat = "json_object",
        client: Any | None = None,
        token_provider: Callable[[], str] | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.deployment = deployment
        self.api = api
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.auth_mode = auth_mode
        self.azure_scope = azure_scope
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.response_format = response_format
        self._client = client
        self._token_provider = token_provider

    @property
    def model_id(self) -> str:
        return f"openai-compatible:{self.deployment}:{self.api}:{self.response_format}"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        return await asyncio.to_thread(self._generate_sync, request)

    def _generate_sync(self, request: GenerationRequest) -> GenerationResponse:
        if self.api == "chat_completions":
            return self._generate_chat_completion(request)
        if self.api == "responses":
            return self._generate_response(request)
        raise ValueError(f"unsupported generator API: {self.api}")

    def _generate_chat_completion(self, request: GenerationRequest) -> GenerationResponse:
        if self.response_format == "structured":
            return self._generate_parsed_chat_completion(request)

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
                "api": "chat_completions",
                "deployment": self.deployment,
                "model": _get(completion, "model"),
                "response_id": _get(completion, "id"),
                "finish_reason": _get(choice, "finish_reason"),
                "prompt_name": request.prompt_name,
            },
        )

    def _generate_parsed_chat_completion(self, request: GenerationRequest) -> GenerationResponse:
        response_model = _required_response_model(request)
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "messages": [{"role": "user", "content": request.prompt}],
            "response_format": response_model,
        }
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens

        completion = self._openai_client().beta.chat.completions.parse(**kwargs)
        choice = _first_choice(completion)
        duration_ms = (time.perf_counter() - started) * 1000.0
        usage = _usage_record(completion, duration_ms=duration_ms)

        return GenerationResponse(
            content=_parsed_chat_content(choice),
            usage=usage,
            metadata={
                "client": self.__class__.__name__,
                "provider": "openai_compatible",
                "api": "chat_completions",
                "response_format": "structured",
                "deployment": self.deployment,
                "model": _get(completion, "model"),
                "response_id": _get(completion, "id"),
                "finish_reason": _get(choice, "finish_reason"),
                "prompt_name": request.prompt_name,
            },
        )

    def _generate_response(self, request: GenerationRequest) -> GenerationResponse:
        if self.response_format == "structured":
            return self._generate_parsed_response(request)

        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "input": [{"role": "user", "content": request.prompt}],
        }
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        if self.response_format == "json_object":
            kwargs["text"] = {"format": {"type": "json_object"}}
        elif self.response_format == "text":
            kwargs["text"] = {"format": {"type": "text"}}

        response = self._openai_client().responses.create(**kwargs)
        duration_ms = (time.perf_counter() - started) * 1000.0
        _raise_for_response_error(response)

        return GenerationResponse(
            content=_response_output_text(response),
            usage=_usage_record(response, duration_ms=duration_ms),
            metadata={
                "client": self.__class__.__name__,
                "provider": "openai_compatible",
                "api": "responses",
                "deployment": self.deployment,
                "model": _get(response, "model"),
                "response_id": _get(response, "id"),
                "status": _get(response, "status"),
                "incomplete_reason": _incomplete_reason(response),
                "prompt_name": request.prompt_name,
            },
        )

    def _generate_parsed_response(self, request: GenerationRequest) -> GenerationResponse:
        response_model = _required_response_model(request)
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "input": [{"role": "user", "content": request.prompt}],
            "text_format": response_model,
        }
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens

        response = self._openai_client().responses.parse(**kwargs)
        duration_ms = (time.perf_counter() - started) * 1000.0
        _raise_for_response_error(response)

        return GenerationResponse(
            content=_parsed_response_content(response),
            usage=_usage_record(response, duration_ms=duration_ms),
            metadata={
                "client": self.__class__.__name__,
                "provider": "openai_compatible",
                "api": "responses",
                "response_format": "structured",
                "deployment": self.deployment,
                "model": _get(response, "model"),
                "response_id": _get(response, "id"),
                "status": _get(response, "status"),
                "incomplete_reason": _incomplete_reason(response),
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
            "max_retries": self.max_retries,
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


def _parsed_chat_content(choice: Any) -> str:
    message = _get(choice, "message")
    parsed = _get(message, "parsed")
    return _parsed_content(parsed)


def _response_output_text(response: Any) -> str:
    output_text = _get(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    text_parts: list[str] = []
    for item in _get(response, "output", []):
        if _get(item, "type") != "message":
            continue
        for content in _get(item, "content", []):
            if _get(content, "type") == "output_text":
                text_parts.append(str(_get(content, "text", "")))

    joined = "".join(text_parts).strip()
    if joined:
        return joined
    raise RuntimeError("model response did not include output text")


def _parsed_response_content(response: Any) -> str:
    return _parsed_content(_get(response, "output_parsed"))


def _parsed_content(parsed: Any) -> str:
    if parsed is None:
        raise RuntimeError("model response did not include parsed structured output")

    to_generation_content = getattr(parsed, "to_generation_content", None)
    if callable(to_generation_content):
        content = to_generation_content()
        if isinstance(content, str) and content.strip():
            return content

    if isinstance(parsed, BaseModel):
        return parsed.model_dump_json()

    return json.dumps(parsed, sort_keys=True)


def _raise_for_response_error(response: Any) -> None:
    error = _get(response, "error")
    if error is not None:
        message = _get(error, "message") or str(error)
        raise RuntimeError(f"model response failed: {message}")

    status = _get(response, "status")
    if status in {"failed", "cancelled", "incomplete"}:
        reason = _incomplete_reason(response) or status
        raise RuntimeError(f"model response ended with status {status}: {reason}")


def _incomplete_reason(response: Any) -> str | None:
    details = _get(response, "incomplete_details")
    reason = _get(details, "reason")
    return str(reason) if reason is not None else None


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


def _required_response_model(request: GenerationRequest) -> type[BaseModel]:
    if request.response_model is None:
        raise RuntimeError("structured generation requires GenerationRequest.response_model")
    return request.response_model


def _normalize_base_url(value: str) -> str:
    split = urlsplit(value.strip())
    path = split.path.rstrip("/")
    for route_suffix in ("/responses", "/chat/completions"):
        if path.endswith(route_suffix):
            path = path[: -len(route_suffix)]
            break
    path = f"{path}/" if path else "/"
    return urlunsplit((split.scheme, split.netloc, path, split.query, split.fragment))
