from __future__ import annotations

from pathlib import Path

from long_document_indexing.config import load_experiment_config
from long_document_indexing.datasets.registry import create_dataset_adapter


def test_enterprise_thin_slice_dataset_has_expected_shape() -> None:
    config = load_experiment_config(
        Path("configs/experiments/enterprise-thin-slice.yaml"),
        project_root=Path.cwd(),
    )
    loaded = create_dataset_adapter(config.dataset).load()
    corpus = loaded.corpora[0]
    question_set = loaded.question_set

    assert corpus.id == "enterprise-thin-slice"
    assert len(corpus.documents) == 8
    assert sum(len(document.segments) for document in corpus.documents) == 24
    assert len(question_set.items) == 16
    assert question_set.capabilities.has_expected_answers is True
    assert question_set.capabilities.has_relevant_documents is True
    assert question_set.capabilities.has_relevant_segments is True


def test_enterprise_thin_slice_gold_labels_reference_existing_sources() -> None:
    config = load_experiment_config(
        Path("configs/experiments/enterprise-thin-slice.yaml"),
        project_root=Path.cwd(),
    )
    loaded = create_dataset_adapter(config.dataset).load()
    corpus = loaded.corpora[0]
    documents_by_id = corpus.document_by_id()
    segments_by_id = corpus.segment_by_id()

    for item in loaded.question_set.items:
        assert item.ground_truth.expected_answer
        assert item.ground_truth.relevant_document_ids
        assert item.ground_truth.relevant_segment_ids
        assert "enterprise_thin_slice" in item.tags

        for document_id in item.ground_truth.relevant_document_ids:
            assert document_id in documents_by_id

        for segment_id in item.ground_truth.relevant_segment_ids:
            assert segment_id in segments_by_id
            assert segments_by_id[segment_id].document_id in item.ground_truth.relevant_document_ids


def test_enterprise_thin_slice_local_config_is_all_system_and_export_enabled() -> None:
    config = load_experiment_config(
        Path("configs/experiments/enterprise-thin-slice.yaml"),
        project_root=Path.cwd(),
    )

    assert config.models.generator_provider == "fake"
    assert config.workflow.runner == "local"
    assert config.answering.mode == "generated"
    assert config.systems == ["flat_vector", "stuffing", "map_reduce", "refine"]
    assert config.evaluation.foundry.enabled is True


def test_enterprise_advanced_thin_slice_config_lists_all_implemented_systems() -> None:
    config = load_experiment_config(
        Path("configs/experiments/enterprise-advanced-thin-slice.yaml"),
        project_root=Path.cwd(),
    )

    assert config.models.generator_provider == "fake"
    assert config.answering.mode == "generated"
    assert config.systems == [
        "flat_vector",
        "stuffing",
        "map_reduce",
        "refine",
        "hierarchical_map",
        "outline_then_fill",
        "agentic_map",
    ]
    assert config.system_config_for("hierarchical_map").hierarchy_branching_factor == 2
    assert config.system_config_for("outline_then_fill").outline_depth == 2
    assert config.system_config_for("agentic_map").agent_target_coverage == 1.0
    assert config.evaluation.foundry.enabled is True


def test_foundry_enterprise_stuffing_config_caps_live_thin_slice_scope(monkeypatch) -> None:
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
        Path("configs/experiments/foundry-enterprise-stuffing-thin-slice.yaml"),
        project_root=Path.cwd(),
    )

    assert config.models.generator_provider == "foundry"
    assert config.models.generator_api == "responses"
    assert config.models.generator_response_format == "structured"
    assert config.models.generator_max_output_tokens == 6000
    assert config.answering.mode == "generated"
    assert config.systems == ["stuffing"]
    assert config.run_control.resume is True
    assert config.run_control.max_model_calls == 24
    assert config.evaluation.foundry.enabled is True
