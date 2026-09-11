from __future__ import annotations

import json
from pathlib import Path

from long_document_indexing.cli import _evaluate_foundry_managed
from long_document_indexing.config import load_experiment_config


def test_foundry_managed_dry_run_does_not_require_run_records(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("configs/experiments/foundry-multilexsum-legal-rag-qa-smoke.yaml"),
        project_root=Path.cwd(),
    )
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(update={"artifacts_dir": tmp_path / "artifacts"})
        }
    )

    _evaluate_foundry_managed(config, dry_run=True)

    plan_path = (
        config.storage.artifacts_dir
        / config.experiment.id
        / "evaluations/foundry/managed-plan.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert plan["dataset_exists"] is False
    assert plan["manifest_exists"] is False
    assert plan["managed_evaluators"] == [
        "groundedness",
        "relevance",
        "retrieval",
        "document_retrieval",
        "response_completeness",
    ]
    assert plan["managed_execution"] == "parallel"
    assert plan["managed_evaluator_delay_seconds"] == 0.0
    assert plan["evaluator_config"]["retrieval"]["column_mapping"] == {
        "query": "${data.query}",
        "context": "${data.context}",
    }
