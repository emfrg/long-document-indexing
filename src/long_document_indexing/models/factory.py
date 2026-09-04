from __future__ import annotations

from long_document_indexing.config import ModelConfig
from long_document_indexing.models.base import TextGenerationClient
from long_document_indexing.models.fake import FakeTextGenerationClient
from long_document_indexing.models.openai_compatible import OpenAICompatibleTextGenerationClient


def create_text_generation_client(config: ModelConfig) -> TextGenerationClient:
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
                config.generator_deployment,
                field_name="generator_deployment",
                provider=config.generator_provider,
            ),
            api_key_env=config.generator_api_key_env,
            auth_mode=config.generator_auth_mode,
            azure_scope=config.generator_azure_scope,
            temperature=config.generator_temperature,
            max_output_tokens=config.generator_max_output_tokens,
            timeout_seconds=config.generator_timeout_seconds,
            response_format=config.generator_response_format,
        )

    raise ValueError(f"unsupported generator provider: {config.generator_provider}")


def _required_resolved_value(value: str | None, *, field_name: str, provider: str) -> str:
    if value is None or not value.strip() or value.strip().startswith("${"):
        raise ValueError(f"models.{field_name} is required when generator_provider={provider}")
    return value.strip()
