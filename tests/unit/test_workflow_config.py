from __future__ import annotations

import pytest

from long_document_indexing.cli import _workflow_runner
from long_document_indexing.config import ExperimentConfig, WorkflowConfig
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
