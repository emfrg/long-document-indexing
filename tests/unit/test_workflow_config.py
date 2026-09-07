from __future__ import annotations

import pytest
from pydantic import ValidationError

from long_document_indexing.cli import _workflow_runner
from long_document_indexing.config import (
    AnsweringConfig,
    ExperimentConfig,
    FoundryEvaluationConfig,
    RunControlConfig,
    WorkflowConfig,
)
from long_document_indexing.workflows.execution import LocalWorkflowRunner, MafWorkflowRunner


def test_workflow_config_defaults_to_local_runner() -> None:
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"id": "exp"},
            "dataset": {"adapter": "jsonl"},
            "systems": ["flat_vector"],
        }
    )

    assert config.workflow.runner == "local"
    assert config.answering.mode == "extractive"
    assert config.run_control.resume is False
    assert config.run_control.force is False
    assert config.evaluation.foundry.enabled is False
    assert config.evaluation.foundry.evaluation_level == "turn"
    assert isinstance(_workflow_runner(config), LocalWorkflowRunner)


def test_workflow_config_selects_maf_runner_when_available() -> None:
    pytest.importorskip("agent_framework")
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"id": "exp"},
            "dataset": {"adapter": "jsonl"},
            "workflow": {"runner": "maf"},
            "systems": ["stuffing"],
        }
    )

    assert isinstance(_workflow_runner(config), MafWorkflowRunner)


def test_workflow_config_rejects_unknown_runner() -> None:
    with pytest.raises(ValueError, match="Input should be"):
        WorkflowConfig.model_validate({"runner": "unknown"})


def test_answering_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Input should be"):
        AnsweringConfig.model_validate({"mode": "unknown"})


def test_run_control_config_rejects_invalid_budget_policy() -> None:
    with pytest.raises(ValidationError, match="cannot both be true"):
        RunControlConfig.model_validate({"resume": True, "force": True})

    with pytest.raises(ValidationError, match="must not be negative"):
        RunControlConfig.model_validate({"max_model_calls": -1})


def test_foundry_evaluation_config_rejects_non_artifact_relative_paths() -> None:
    with pytest.raises(ValidationError, match="artifact-relative"):
        FoundryEvaluationConfig.model_validate({"dataset_path": "/tmp/foundry.jsonl"})

    with pytest.raises(ValidationError, match="artifact-relative"):
        FoundryEvaluationConfig.model_validate({"manifest_path": "../foundry.json"})

    with pytest.raises(ValidationError, match="artifact-relative"):
        FoundryEvaluationConfig.model_validate({"result_path": "/tmp/result.json"})


def test_foundry_evaluation_config_rejects_blank_or_duplicate_evaluators() -> None:
    with pytest.raises(ValidationError, match="blank"):
        FoundryEvaluationConfig.model_validate({"evaluators": ["groundedness", " "]})

    with pytest.raises(ValidationError, match="duplicates"):
        FoundryEvaluationConfig.model_validate({"evaluators": ["relevance", "relevance"]})

    with pytest.raises(ValidationError, match="duplicates"):
        FoundryEvaluationConfig.model_validate({"managed_evaluators": ["f1", "f1"]})


def test_foundry_evaluation_config_rejects_blank_project_or_tags() -> None:
    with pytest.raises(ValidationError, match="blank"):
        FoundryEvaluationConfig.model_validate({"azure_ai_project": " "})

    with pytest.raises(ValidationError, match="blank"):
        FoundryEvaluationConfig.model_validate({"tags": {"": "value"}})


def test_foundry_evaluation_config_rejects_conversation_level_for_now() -> None:
    with pytest.raises(ValidationError, match="Input should be 'turn'"):
        FoundryEvaluationConfig.model_validate({"evaluation_level": "conversation"})
