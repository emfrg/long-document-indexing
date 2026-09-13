from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections import Counter
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from long_document_indexing.domain.runs import UsageRecord
from long_document_indexing.models.base import EmbeddingResponse
from long_document_indexing.text import tokenize


class OpenAICompatibleEmbeddingClient:
    """Embedding client for Foundry/Azure OpenAI compatible v1 endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        deployment: str,
        api_key: str | None = None,
        api_key_env: str = "AZURE_INFERENCE_CREDENTIAL",
        auth_mode: str = "api_key",
        azure_scope: str = "https://ai.azure.com/.default",
        timeout_seconds: float = 60.0,
        dimensions: int | None = None,
        client: Any | None = None,
        token_provider: Callable[[], str] | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.deployment = deployment
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.auth_mode = auth_mode
        self.azure_scope = azure_scope
        self.timeout_seconds = timeout_seconds
        self.dimensions = dimensions
        self._client = client
        self._token_provider = token_provider

    @property
    def model_id(self) -> str:
        suffix = f":{self.dimensions}" if self.dimensions is not None else ""
        return f"openai-compatible:{self.deployment}{suffix}"

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        if not texts:
            return EmbeddingResponse(embeddings=[])
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> EmbeddingResponse:
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "input": texts,
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions

        response = self._openai_client().embeddings.create(**kwargs)
        duration_ms = (time.perf_counter() - started) * 1000.0
        data = sorted(_get(response, "data", []), key=lambda item: int(_get(item, "index", 0)))
        embeddings = [[float(value) for value in _get(item, "embedding", [])] for item in data]
        _validate_embeddings(embeddings, expected_count=len(texts))
        usage = _get(response, "usage")
        input_tokens = _int_field(usage, "prompt_tokens", "input_tokens", "total_tokens")
        return EmbeddingResponse(
            embeddings=embeddings,
            usage=UsageRecord(
                input_tokens=input_tokens,
                model_calls=1,
                duration_ms=duration_ms,
            ),
            metadata={
                "client": self.__class__.__name__,
                "provider": "openai_compatible",
                "deployment": self.deployment,
                "model": _get(response, "model"),
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
                "Dense retrieval requires the optional `foundry` extra. "
                "Install it with `uv sync --extra foundry`."
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
            raise ValueError(f"unsupported embedding auth mode: {self.auth_mode}")
        return OpenAI(**kwargs)

    def _required_api_key(self) -> str:
        if self.api_key:
            return self.api_key

        import os

        value = os.environ.get(self.api_key_env)
        if value:
            return value
        raise RuntimeError(
            f"missing API key for embeddings; set environment variable {self.api_key_env}"
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


class FakeEmbeddingClient:
    """Deterministic hashed-token embeddings for tests and offline smoke runs."""

    def __init__(self, *, dimensions: int = 256) -> None:
        if dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        self.dimensions = dimensions

    @property
    def model_id(self) -> str:
        return f"fake-hashed-token-v1:{self.dimensions}"

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        embeddings = [_hashed_embedding(text, dimensions=self.dimensions) for text in texts]
        return EmbeddingResponse(
            embeddings=embeddings,
            usage=UsageRecord(
                input_tokens=sum(len(tokenize(text)) for text in texts),
                model_calls=1 if texts else 0,
            ),
            metadata={"client": self.__class__.__name__, "model": self.model_id},
        )


def _hashed_embedding(text: str, *, dimensions: int) -> list[float]:
    counts = Counter(tokenize(text))
    vector = [0.0] * dimensions
    for token, count in counts.items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % dimensions
        sign = 1.0 if digest[8] % 2 == 0 else -1.0
        vector[index] += sign * float(count)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _validate_embeddings(embeddings: list[list[float]], *, expected_count: int) -> None:
    if len(embeddings) != expected_count:
        raise RuntimeError(
            f"embedding response count mismatch: expected {expected_count}, got {len(embeddings)}"
        )
    if not embeddings:
        return
    dimensions = len(embeddings[0])
    if dimensions < 1:
        raise RuntimeError("embedding response contained an empty vector")
    if any(len(embedding) != dimensions for embedding in embeddings):
        raise RuntimeError("embedding response contained inconsistent vector dimensions")
    if any(not math.isfinite(value) for embedding in embeddings for value in embedding):
        raise RuntimeError("embedding response contained a non-finite value")


def _normalize_base_url(value: str) -> str:
    split = urlsplit(value.strip())
    path = split.path.rstrip("/")
    for route_suffix in ("/embeddings", "/responses", "/chat/completions"):
        if path.endswith(route_suffix):
            path = path[: -len(route_suffix)]
            break
    path = f"{path}/" if path else "/"
    return urlunsplit((split.scheme, split.netloc, path, split.query, split.fragment))


def _get(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, *names: str) -> int:
    for name in names:
        raw = _get(value, name)
        if raw is not None:
            return int(raw)
    return 0
