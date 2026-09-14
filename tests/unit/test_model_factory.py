from __future__ import annotations

from pathlib import Path

import pytest

from long_document_indexing.config import ModelConfig, load_experiment_config
from long_document_indexing.models.embeddings import OpenAICompatibleEmbeddingClient
from long_document_indexing.models.factory import (
    create_embedding_client,
    create_text_generation_client,
)
from long_document_indexing.models.fake import FakeTextGenerationClient
from long_document_indexing.models.openai_compatible import OpenAICompatibleTextGenerationClient


def test_model_factory_defaults_to_fake_client() -> None:
    client = create_text_generation_client(ModelConfig())

    assert isinstance(client, FakeTextGenerationClient)


def test_model_factory_creates_foundry_client_without_importing_openai() -> None:
    client = create_text_generation_client(
        ModelConfig(
            generator_provider="foundry",
            generator_base_url="https://example.openai.azure.com/openai/v1/",
            generator_deployment="doc-map",
            generator_api="responses",
            generator_max_retries=9,
            generator_response_format="structured",
        )
    )

    assert isinstance(client, OpenAICompatibleTextGenerationClient)
    assert client.deployment == "doc-map"
    assert client.api == "responses"
    assert client.max_retries == 9
    assert client.response_format == "structured"


def test_model_factory_configures_token_safe_embedding_client() -> None:
    client = create_embedding_client(
        ModelConfig(
            generator_provider="foundry",
            generator_base_url="https://example.openai.azure.com/openai/v1/",
            embedding_deployment="text-embedding-3-large",
            embedding_max_input_tokens=8000,
            embedding_max_batch_tokens=200_000,
            embedding_max_retries=9,
        )
    )

    assert isinstance(client, OpenAICompatibleEmbeddingClient)
    assert client.max_input_tokens == 8000
    assert client.max_batch_tokens == 200_000
    assert client.max_retries == 9


def test_model_factory_selects_explicit_role_deployments_with_generator_fallback() -> None:
    config = ModelConfig(
        generator_provider="foundry",
        generator_base_url="https://example.openai.azure.com/openai/v1/",
        generator_deployment="map-builder",
        router_deployment="map-router",
        answer_deployment="rag-answerer",
    )

    assert create_text_generation_client(config, role="generator").deployment == "map-builder"
    assert create_text_generation_client(config, role="router").deployment == "map-router"
    assert create_text_generation_client(config, role="answer").deployment == "rag-answerer"

    fallback = config.model_copy(update={"router_deployment": "", "answer_deployment": None})
    assert create_text_generation_client(fallback, role="router").deployment == "map-builder"
    assert create_text_generation_client(fallback, role="answer").deployment == "map-builder"


def test_model_factory_requires_resolved_real_client_values() -> None:
    with pytest.raises(ValueError, match="generator_base_url"):
        create_text_generation_client(
            ModelConfig(
                generator_provider="foundry",
                generator_base_url="${FOUNDRY_GENERATOR_BASE_URL}",
                generator_deployment="doc-map",
            )
        )


def test_model_config_loads_real_client_values_from_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FOUNDRY_GENERATOR_BASE_URL", "https://example.openai.azure.com/openai/v1/")
    monkeypatch.setenv("FOUNDRY_GENERATOR_DEPLOYMENT", "doc-map")
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
experiment:
  id: env-config
dataset:
  adapter: jsonl
models:
  generator_provider: foundry
  generator_api: responses
  generator_response_format: structured
  generator_base_url: ${FOUNDRY_GENERATOR_BASE_URL}
  generator_deployment: ${FOUNDRY_GENERATOR_DEPLOYMENT}
systems:
  - stuffing
""".strip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path, project_root=tmp_path)

    assert config.models.generator_provider == "foundry"
    assert config.models.generator_api == "responses"
    assert config.models.generator_response_format == "structured"
    assert config.models.generator_base_url == "https://example.openai.azure.com/openai/v1/"
    assert config.models.generator_deployment == "doc-map"


def test_model_config_loads_values_from_dotenv(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FOUNDRY_GENERATOR_BASE_URL", raising=False)
    monkeypatch.delenv("FOUNDRY_GENERATOR_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_INFERENCE_CREDENTIAL", raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "FOUNDRY_GENERATOR_BASE_URL=https://example.openai.azure.com/openai/v1/responses",
                "FOUNDRY_GENERATOR_DEPLOYMENT=gpt-5-mini-doc-map-generator",
                "AZURE_INFERENCE_CREDENTIAL=test-key",
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
experiment:
  id: dotenv-config
dataset:
  adapter: jsonl
models:
  generator_provider: foundry
  generator_api: responses
  generator_response_format: structured
  generator_base_url: ${FOUNDRY_GENERATOR_BASE_URL}
  generator_deployment: ${FOUNDRY_GENERATOR_DEPLOYMENT}
systems:
  - stuffing
""".strip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path, project_root=tmp_path)

    assert config.models.generator_base_url.endswith("/responses")
    assert config.models.generator_deployment == "gpt-5-mini-doc-map-generator"
    assert config.models.generator_response_format == "structured"
    assert create_text_generation_client(config.models).base_url.endswith("/openai/v1/")


def test_foundry_multi_system_real_smoke_config_shape(monkeypatch) -> None:
    monkeypatch.setenv(
        "FOUNDRY_GENERATOR_BASE_URL",
        "https://example.services.ai.azure.com/openai/v1/responses",
    )
    monkeypatch.setenv("FOUNDRY_GENERATOR_DEPLOYMENT", "gpt-5-mini-doc-map-generator")
    monkeypatch.setenv("FOUNDRY_GENERATOR_API", "responses")
    monkeypatch.setenv("FOUNDRY_GENERATOR_MAX_OUTPUT_TOKENS", "6000")
    monkeypatch.setenv("FOUNDRY_GENERATOR_RESPONSE_FORMAT", "structured")
    monkeypatch.setenv("AZURE_INFERENCE_CREDENTIAL", "test-key")

    config = load_experiment_config(
        Path("configs/experiments/foundry-multi-system-real-smoke.yaml"),
        project_root=Path.cwd(),
    )

    assert config.experiment.id == "foundry-multi-system-real-smoke"
    assert config.models.generator_provider == "foundry"
    assert config.models.generator_api == "responses"
    assert config.models.generator_max_output_tokens == 6000
    assert config.models.generator_response_format == "structured"
    assert config.answering.mode == "generated"
    assert config.systems == ["flat_vector", "stuffing", "map_reduce", "refine"]
    assert config.run_control.resume is True
    assert config.run_control.max_model_calls == 20
    assert config.evaluation.foundry.enabled is True
    assert config.evaluation.foundry.evaluation_level == "turn"
