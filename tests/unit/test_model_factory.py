from __future__ import annotations

import pytest

from long_document_indexing.config import ModelConfig, load_experiment_config
from long_document_indexing.models.factory import create_text_generation_client
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
        )
    )

    assert isinstance(client, OpenAICompatibleTextGenerationClient)
    assert client.deployment == "doc-map"


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
  generator_base_url: ${FOUNDRY_GENERATOR_BASE_URL}
  generator_deployment: ${FOUNDRY_GENERATOR_DEPLOYMENT}
systems:
  - stuffing
""".strip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path, project_root=tmp_path)

    assert config.models.generator_provider == "foundry"
    assert config.models.generator_base_url == "https://example.openai.azure.com/openai/v1/"
    assert config.models.generator_deployment == "doc-map"
