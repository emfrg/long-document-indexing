from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from long_document_indexing.config import FoundryEvaluationConfig, load_experiment_config
from long_document_indexing.domain.benchmark import BenchmarkItem, GroundTruth
from long_document_indexing.domain.runs import Citation, RagRunRecord, RetrievedItem, UsageRecord
from long_document_indexing.evaluation.foundry import (
    build_foundry_managed_evaluation_plan,
    run_foundry_managed_evaluation,
    write_foundry_evaluation_export,
)
from long_document_indexing.storage.artifacts import ArtifactStore


def test_foundry_managed_evaluation_plan_uses_export_paths_and_env_project(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "FOUNDRY_EVALUATION_PROJECT_ENDPOINT",
        "https://example.services.ai.azure.com/api/projects/project-a",
    )
    config = _config(tmp_path)
    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)
    _write_export(store, config.evaluation.foundry)

    plan = build_foundry_managed_evaluation_plan(config=config, store=store)

    assert plan["evaluation_name"] == "foundry-eval-export-smoke"
    assert plan["dataset_exists"] is True
    assert plan["manifest_exists"] is True
    assert (
        plan["azure_ai_project"] == "https://example.services.ai.azure.com/api/projects/project-a"
    )
    assert plan["managed_evaluators"] == ["f1", "rouge"]
    assert plan["evaluator_config"]["f1"]["column_mapping"] == {
        "response": "${data.response}",
        "ground_truth": "${data.ground_truth}",
    }


def test_foundry_managed_evaluation_calls_injected_evaluate_and_writes_result(tmp_path) -> None:
    config = _config(tmp_path)
    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)
    _write_export(store, config.evaluation.foundry)
    calls: list[dict[str, Any]] = []

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "metrics": {"f1": 1.0, "rouge": 0.75},
            "rows": [{"id": "run-alpha"}],
            "studio_url": "https://ai.azure.com/evaluations/run-alpha",
            "oai_eval_run_ids": [{"id": "eval-run"}],
        }

    result = run_foundry_managed_evaluation(
        config=config,
        store=store,
        evaluate_fn=fake_evaluate,
    )

    assert calls[0]["data"].endswith("evaluations/foundry/dataset.jsonl")
    assert sorted(calls[0]["evaluators"]) == ["f1", "rouge"]
    assert calls[0]["evaluation_name"] == "foundry-eval-export-smoke"
    assert calls[0]["evaluator_config"]["rouge"]["column_mapping"]["response"] == (
        "${data.response}"
    )
    assert result.metrics == {"f1": 1.0, "rouge": 0.75}
    assert result.row_count == 1
    assert result.studio_url == "https://ai.azure.com/evaluations/run-alpha"
    assert (store.experiment_dir / config.evaluation.foundry.result_path).exists()


def test_foundry_managed_evaluation_requires_exported_dataset(tmp_path) -> None:
    config = _config(tmp_path)
    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)

    with pytest.raises(FileNotFoundError, match="Run `ldi evaluate`"):
        run_foundry_managed_evaluation(
            config=config,
            store=store,
            evaluate_fn=lambda **_: {},
        )


def _config(tmp_path):
    config = load_experiment_config(
        Path("configs/experiments/foundry-eval-export-smoke.yaml"),
        project_root=Path.cwd(),
    )
    return config.model_copy(
        update={
            "storage": config.storage.model_copy(update={"artifacts_dir": tmp_path / "artifacts"})
        }
    )


def _write_export(store: ArtifactStore, config: FoundryEvaluationConfig) -> None:
    write_foundry_evaluation_export(
        store=store,
        records=[_record()],
        items_by_id={"q_alpha": _item()},
        config=config,
        experiment_id="foundry-eval-export-smoke",
    )


def _item() -> BenchmarkItem:
    return BenchmarkItem(
        id="q_alpha",
        corpus_id="smoke-corpus",
        query="What did the board approve?",
        ground_truth=GroundTruth(
            expected_answer="The board approved the Alpha renewal.",
            relevant_document_ids={"doc_alpha"},
            relevant_segment_ids={"alpha_s1"},
        ),
    )


def _record() -> RagRunRecord:
    return RagRunRecord(
        run_id="run-alpha",
        experiment_id="foundry-eval-export-smoke",
        system_id="stuffing",
        corpus_id="smoke-corpus",
        item_id="q_alpha",
        selected_document_ids=["doc_alpha"],
        retrieved_items=[
            RetrievedItem(
                document_id="doc_alpha",
                segment_id="alpha_s1",
                text="The board approved the Alpha renewal.",
                score=0.98,
                rank=1,
                retrieval_stage="map",
            )
        ],
        answer="The board approved the Alpha renewal.",
        citations=[
            Citation(
                document_id="doc_alpha",
                segment_id="alpha_s1",
                quote="The board approved the Alpha renewal.",
            )
        ],
        usage=UsageRecord(input_tokens=10, output_tokens=12, model_calls=1, tool_calls=1),
        status="succeeded",
    )
