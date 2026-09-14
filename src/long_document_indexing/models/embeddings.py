from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
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
        max_input_tokens: int = 8191,
        max_batch_tokens: int = 250_000,
        max_retries: int = 12,
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
        self.max_input_tokens = max_input_tokens
        self.max_batch_tokens = max_batch_tokens
        self.max_retries = max_retries
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
        chunks = _chunk_embedding_inputs(
            texts,
            deployment=self.deployment,
            max_input_tokens=self.max_input_tokens,
        )
        chunk_embeddings: list[list[float]] = []
        input_tokens = 0
        model_calls = 0
        response_model: Any = None
        for batch in _embedding_batches(chunks, max_batch_tokens=self.max_batch_tokens):
            kwargs: dict[str, Any] = {
                "model": self.deployment,
                "input": [chunk.text for chunk in batch],
                "encoding_format": "float",
            }
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions

            response = self._openai_client().embeddings.create(**kwargs)
            data = sorted(
                _get(response, "data", []),
                key=lambda item: int(_get(item, "index", 0)),
            )
            batch_embeddings = [
                [float(value) for value in _get(item, "embedding", [])] for item in data
            ]
            _validate_embeddings(batch_embeddings, expected_count=len(batch))
            chunk_embeddings.extend(batch_embeddings)
            input_tokens += _int_field(
                _get(response, "usage"),
                "prompt_tokens",
                "input_tokens",
                "total_tokens",
            )
            model_calls += 1
            response_model = _get(response, "model", response_model)

        duration_ms = (time.perf_counter() - started) * 1000.0
        embeddings = _pool_chunk_embeddings(
            chunks,
            chunk_embeddings,
            expected_count=len(texts),
        )
        _validate_embeddings(embeddings, expected_count=len(texts))
        chunk_counts = Counter(chunk.input_index for chunk in chunks)
        return EmbeddingResponse(
            embeddings=embeddings,
            usage=UsageRecord(
                input_tokens=input_tokens,
                model_calls=model_calls,
                duration_ms=duration_ms,
            ),
            metadata={
                "client": self.__class__.__name__,
                "provider": "openai_compatible",
                "deployment": self.deployment,
                "model": response_model,
                "input_count": len(texts),
                "embedded_chunk_count": len(chunks),
                "split_input_count": sum(count > 1 for count in chunk_counts.values()),
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
            "max_retries": self.max_retries,
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


@dataclass(frozen=True)
class _EmbeddingChunk:
    input_index: int
    text: str
    token_count: int


def _chunk_embedding_inputs(
    texts: list[str],
    *,
    deployment: str,
    max_input_tokens: int,
) -> list[_EmbeddingChunk]:
    if max_input_tokens < 1:
        raise ValueError("embedding max input tokens must be positive")
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "token-safe Foundry embeddings require the optional `foundry` extra"
        ) from exc

    try:
        encoding = tiktoken.encoding_for_model(deployment)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    chunks: list[_EmbeddingChunk] = []
    for input_index, value in enumerate(texts):
        token_ids = encoding.encode(value, disallowed_special=())
        if not token_ids:
            chunks.append(_EmbeddingChunk(input_index=input_index, text=value, token_count=1))
            continue
        for start in range(0, len(token_ids), max_input_tokens):
            chunk_ids = token_ids[start : start + max_input_tokens]
            chunks.append(
                _EmbeddingChunk(
                    input_index=input_index,
                    text=encoding.decode(chunk_ids),
                    token_count=len(chunk_ids),
                )
            )
    return chunks


def _embedding_batches(
    chunks: list[_EmbeddingChunk],
    *,
    max_batch_tokens: int,
) -> list[list[_EmbeddingChunk]]:
    if max_batch_tokens < 1:
        raise ValueError("embedding max batch tokens must be positive")

    batches: list[list[_EmbeddingChunk]] = []
    batch: list[_EmbeddingChunk] = []
    batch_tokens = 0
    for chunk in chunks:
        if chunk.token_count > max_batch_tokens:
            raise ValueError("embedding chunk exceeds the configured batch token limit")
        if batch and batch_tokens + chunk.token_count > max_batch_tokens:
            batches.append(batch)
            batch = []
            batch_tokens = 0
        batch.append(chunk)
        batch_tokens += chunk.token_count
    if batch:
        batches.append(batch)
    return batches


def _pool_chunk_embeddings(
    chunks: list[_EmbeddingChunk],
    chunk_embeddings: list[list[float]],
    *,
    expected_count: int,
) -> list[list[float]]:
    _validate_embeddings(chunk_embeddings, expected_count=len(chunks))
    grouped: list[list[tuple[list[float], int]]] = [[] for _ in range(expected_count)]
    for chunk, embedding in zip(chunks, chunk_embeddings, strict=True):
        grouped[chunk.input_index].append((embedding, chunk.token_count))

    pooled: list[list[float]] = []
    for values in grouped:
        if not values:
            raise RuntimeError("embedding chunks did not cover every input")
        if len(values) == 1:
            pooled.append(values[0][0])
            continue
        total_weight = sum(weight for _, weight in values)
        dimensions = len(values[0][0])
        average = [
            sum(embedding[index] * weight for embedding, weight in values) / total_weight
            for index in range(dimensions)
        ]
        norm = math.sqrt(sum(value * value for value in average))
        pooled.append([value / norm for value in average] if norm else average)
    return pooled


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
