from __future__ import annotations

from typing import Literal

from long_document_indexing.config import ModelConfig
from long_document_indexing.models.base import EmbeddingClient, TextGenerationClient
from long_document_indexing.models.embeddings import (
    FakeEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
)
from long_document_indexing.models.fake import FakeTextGenerationClient
from long_document_indexing.models.openai_compatible import (
    OpenAICompatibleTextGenerationClient,
)

GenerationRole = Literal["generator", "router", "answer"]


def create_text_generation_client(
    config: ModelConfig,
    *,
    role: GenerationRole = "generator",
) -> TextGenerationClient:
    if config.generator_provider == "fake":
        return FakeTextGenerationClient()

    if config.generator_provider in {"foundry", "openai_compatible"}:
        return OpenAICompatibleTextGenerationClient(
            base_url=_required_resolved_value(
                config.generator_base_url,
                field_name="generator_base_url",
                provider=config.generator_provider,
            ),
            deployment=_required_resolved_value(
                generation_deployment_for_role(config, role),
                field_name=f"{role}_deployment or generator_deployment",
                provider=config.generator_provider,
            ),
            api=config.generator_api,
            api_key_env=config.generator_api_key_env,
            auth_mode=config.generator_auth_mode,
            azure_scope=config.generator_azure_scope,
            temperature=config.generator_temperature,
            max_output_tokens=config.generator_max_output_tokens,
            timeout_seconds=config.generator_timeout_seconds,
            response_format=config.generator_response_format,
        )

    raise ValueError(f"unsupported generator provider: {config.generator_provider}")


def generation_deployment_for_role(
    config: ModelConfig,
    role: GenerationRole,
) -> str | None:
    role_deployment = {
        "generator": config.generator_deployment,
        "router": config.router_deployment,
        "answer": config.answer_deployment,
    }[role]
    return _optional_resolved_value(role_deployment) or _optional_resolved_value(
        config.generator_deployment
    )


def create_embedding_client(config: ModelConfig) -> EmbeddingClient:
    provider = config.embedding_provider or config.generator_provider
    if provider == "fake":
        return FakeEmbeddingClient(dimensions=config.embedding_dimensions or 256)

    if provider in {"foundry", "openai_compatible"}:
        return OpenAICompatibleEmbeddingClient(
            base_url=_required_resolved_value(
                config.embedding_base_url or config.generator_base_url,
                field_name="embedding_base_url or generator_base_url",
                provider=provider,
            ),
            deployment=_required_resolved_value(
                config.embedding_deployment,
                field_name="embedding_deployment",
                provider=provider,
            ),
            api_key_env=config.embedding_api_key_env or config.generator_api_key_env,
            auth_mode=config.embedding_auth_mode or config.generator_auth_mode,
            azure_scope=config.embedding_azure_scope or config.generator_azure_scope,
            timeout_seconds=config.embedding_timeout_seconds,
            dimensions=config.embedding_dimensions,
        )

    raise ValueError(f"unsupported embedding provider: {provider}")


def _required_resolved_value(value: str | None, *, field_name: str, provider: str) -> str:
    if value is None or not value.strip() or value.strip().startswith("${"):
        raise ValueError(f"models.{field_name} is required when provider={provider}")
    return value.strip()


def _optional_resolved_value(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.startswith("${"):
        return None
    return stripped
