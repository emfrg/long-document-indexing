from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from long_document_indexing.config import (
    ExperimentConfig,
    FoundryEvaluationConfig,
    load_experiment_config,
)
from long_document_indexing.domain.benchmark import BenchmarkItem, GroundTruth
from long_document_indexing.domain.runs import Citation, RagRunRecord, RetrievedItem, UsageRecord
from long_document_indexing.evaluation.foundry import (
    build_foundry_managed_evaluation_plan,
    run_foundry_managed_evaluation,
    write_foundry_evaluation_export,
)
from long_document_indexing.evaluation.foundry.managed import (
    _is_reasoning_model_deployment,
    _managed_evaluator_config,
    _managed_evaluators,
    _resolved_azure_openai_endpoint,
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
    assert plan["managed_execution"] == "parallel"
    assert plan["managed_group_by"] == "none"
    assert plan["managed_sample_fraction"] == 1.0
    assert plan["source_row_count"] == 1
    assert plan["selected_item_count"] == 1
    assert plan["evaluated_row_count"] == 1
    assert plan["scope_row_counts"] == {"all": 1}
    assert plan["managed_evaluator_delay_seconds"] == 0.0
    assert plan["managed_max_attempts"] == 3
    assert plan["managed_retry_delay_seconds"] == 30.0
    assert plan["judge_model_config"] is None
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
    manifest = store.read_json(config.evaluation.foundry.manifest_path)
    assert result.metadata["dataset_sha256"] == manifest["dataset_sha256"]
    assert (store.experiment_dir / config.evaluation.foundry.result_path).exists()


def test_foundry_managed_evaluation_rejects_dataset_manifest_mismatch(tmp_path) -> None:
    config = _config(tmp_path)
    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)
    _write_export(store, config.evaluation.foundry)
    dataset_path = store.experiment_dir / config.evaluation.foundry.dataset_path
    dataset_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its manifest"):
        run_foundry_managed_evaluation(
            config=config,
            store=store,
            evaluate_fn=lambda **_: {},
        )


def test_foundry_managed_evaluation_rejects_failed_or_empty_rows_before_sdk_call(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)
    failed_record = _record().model_copy(
        update={
            "status": "failed",
            "answer": "",
            "retrieved_items": [],
            "citations": [],
            "error": "answer generation failed",
        }
    )
    write_foundry_evaluation_export(
        store=store,
        records=[failed_record],
        items_by_id={"q_alpha": _item()},
        config=config.evaluation.foundry,
        experiment_id=config.experiment.id,
    )
    calls = 0

    def fake_evaluate(**_: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {}

    with pytest.raises(
        ValueError,
        match=r"stuffing/q_alpha \(status=failed; empty=response\)",
    ):
        run_foundry_managed_evaluation(
            config=config,
            store=store,
            evaluate_fn=fake_evaluate,
        )

    assert calls == 0


def test_foundry_managed_evaluation_calls_injected_evaluate_with_rag_mappings(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    foundry_config = config.evaluation.foundry.model_copy(
        update={
            "managed_evaluators": [
                "groundedness",
                "retrieval",
                "document_retrieval",
                "response_completeness",
            ]
        }
    )
    config = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(update={"foundry": foundry_config})
        }
    )
    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)
    _write_export(store, foundry_config)
    calls: list[dict[str, Any]] = []

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "metrics": {
                "groundedness.gpt_groundedness": 4.0,
                "retrieval.retrieval": 4.0,
                "document_retrieval.document_retrieval": 1.0,
                "response_completeness.response_completeness": 4.0,
            },
            "rows": [{"id": "run-alpha"}],
        }

    result = run_foundry_managed_evaluation(
        config=config,
        store=store,
        evaluate_fn=fake_evaluate,
    )

    assert sorted(calls[0]["evaluators"]) == [
        "document_retrieval",
        "groundedness",
        "response_completeness",
        "retrieval",
    ]
    assert calls[0]["evaluator_config"]["groundedness"]["column_mapping"] == {
        "query": "${data.query}",
        "response": "${data.response}",
        "context": "${data.context}",
    }
    assert calls[0]["evaluator_config"]["document_retrieval"]["column_mapping"] == {
        "retrieval_ground_truth": "${data.retrieval_ground_truth}",
        "retrieved_documents": "${data.retrieved_documents}",
    }
    assert result.metrics == {
        "groundedness.gpt_groundedness": 4.0,
        "retrieval.retrieval": 4.0,
        "document_retrieval.document_retrieval": 1.0,
        "response_completeness.response_completeness": 4.0,
    }


def test_foundry_managed_evaluation_can_run_evaluators_sequentially(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    foundry_config = config.evaluation.foundry.model_copy(
        update={
            "managed_evaluators": ["groundedness", "retrieval"],
            "managed_execution": "sequential",
        }
    )
    config = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(update={"foundry": foundry_config})
        }
    )
    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)
    _write_export(store, foundry_config)
    calls: list[dict[str, Any]] = []

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        evaluator_name = next(iter(kwargs["evaluators"]))
        return {
            "metrics": {f"{evaluator_name}.score": 1.0},
            "rows": [{"id": f"row-{evaluator_name}"}],
            "studio_url": f"https://ai.azure.com/evaluations/{evaluator_name}",
        }

    result = run_foundry_managed_evaluation(
        config=config,
        store=store,
        evaluate_fn=fake_evaluate,
    )

    assert [list(call["evaluators"]) for call in calls] == [["groundedness"], ["retrieval"]]
    assert calls[0]["evaluation_name"] == "foundry-eval-export-smoke-groundedness"
    assert calls[1]["evaluation_name"] == "foundry-eval-export-smoke-retrieval"
    assert calls[0]["tags"]["managed_evaluator"] == "groundedness"
    assert calls[1]["tags"]["managed_evaluator"] == "retrieval"
    assert result.metrics == {
        "groundedness.score": 1.0,
        "retrieval.score": 1.0,
    }
    assert result.row_count == 1
    assert result.studio_url == "https://ai.azure.com/evaluations/groundedness"
    assert result.metadata["managed_execution"] == "sequential"
    assert len(result.metadata["evaluator_results"]) == 2


def test_foundry_managed_evaluation_retries_transient_connection_errors(tmp_path) -> None:
    config = _config(tmp_path)
    foundry_config = config.evaluation.foundry.model_copy(
        update={
            "managed_evaluators": ["groundedness"],
            "managed_max_attempts": 3,
            "managed_retry_delay_seconds": 2.0,
        }
    )
    config = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(update={"foundry": foundry_config})
        }
    )
    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)
    _write_export(store, foundry_config)
    calls = 0
    sleeps: list[float] = []

    def fake_evaluate(**_: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("APIConnectionError: Connection error")
        return {
            "metrics": {"groundedness.groundedness": 4.0},
            "rows": [{"id": "run-alpha"}],
        }

    result = run_foundry_managed_evaluation(
        config=config,
        store=store,
        evaluate_fn=fake_evaluate,
        sleep_fn=sleeps.append,
    )

    assert calls == 2
    assert sleeps == [2.0]
    assert result.metadata["attempts"] == 2
    assert result.metadata["completion_status"] == "complete"


def test_foundry_managed_evaluation_rejects_partial_results_after_retries(tmp_path) -> None:
    config = _config(tmp_path)
    foundry_config = config.evaluation.foundry.model_copy(
        update={
            "managed_evaluators": ["groundedness"],
            "managed_max_attempts": 2,
            "managed_retry_delay_seconds": 0.0,
        }
    )
    config = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(update={"foundry": foundry_config})
        }
    )
    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)
    _write_export(store, foundry_config)
    calls = 0

    def fake_evaluate(**_: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"metrics": {"groundedness.groundedness": 4.0}, "rows": []}

    with pytest.raises(RuntimeError, match="did not complete after 2 attempt"):
        run_foundry_managed_evaluation(
            config=config,
            store=store,
            evaluate_fn=fake_evaluate,
            sleep_fn=lambda _: None,
        )

    assert calls == 2
    assert not (store.experiment_dir / foundry_config.result_path).exists()


def test_sequential_managed_evaluation_reuses_completed_evaluator_checkpoint(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    foundry_config = config.evaluation.foundry.model_copy(
        update={
            "managed_evaluators": ["groundedness", "retrieval"],
            "managed_execution": "sequential",
            "managed_max_attempts": 1,
        }
    )
    config = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(update={"foundry": foundry_config})
        }
    )
    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)
    _write_export(store, foundry_config)
    first_calls: list[str] = []

    def interrupted_evaluate(**kwargs: Any) -> dict[str, Any]:
        evaluator_name = next(iter(kwargs["evaluators"]))
        first_calls.append(evaluator_name)
        if evaluator_name == "retrieval":
            raise RuntimeError("APIConnectionError: Connection error")
        return {
            "metrics": {"groundedness.groundedness": 4.0},
            "rows": [{"id": "run-alpha"}],
        }

    with pytest.raises(RuntimeError, match="retrieval did not complete"):
        run_foundry_managed_evaluation(
            config=config,
            store=store,
            evaluate_fn=interrupted_evaluate,
            sleep_fn=lambda _: None,
        )

    assert first_calls == ["groundedness", "retrieval"]
    second_calls: list[str] = []

    def resumed_evaluate(**kwargs: Any) -> dict[str, Any]:
        evaluator_name = next(iter(kwargs["evaluators"]))
        second_calls.append(evaluator_name)
        return {
            "metrics": {"retrieval.retrieval": 4.0},
            "rows": [{"id": "run-alpha"}],
        }

    result = run_foundry_managed_evaluation(
        config=config,
        store=store,
        evaluate_fn=resumed_evaluate,
        sleep_fn=lambda _: None,
    )

    assert second_calls == ["retrieval"]
    assert result.metadata["completion_status"] == "complete"
    assert result.metadata["evaluator_results"][0]["reused"] is True
    assert result.metadata["evaluator_results"][1]["reused"] is False


def test_managed_evaluation_samples_same_items_and_reports_each_system(tmp_path) -> None:
    config = _config(tmp_path)
    foundry_config = config.evaluation.foundry.model_copy(
        update={
            "managed_evaluators": ["groundedness"],
            "managed_execution": "sequential",
            "managed_group_by": "system_id",
            "managed_sample_fraction": 0.5,
            "managed_sample_seed": 7,
        }
    )
    config = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(update={"foundry": foundry_config})
        }
    )
    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)
    items = {
        f"q-{question_type}-{index}": BenchmarkItem(
            id=f"q-{question_type}-{index}",
            corpus_id="smoke-corpus",
            query=f"Question {question_type} {index}?",
            ground_truth=GroundTruth(expected_answer="Expected."),
            tags={question_type},
        )
        for question_type in ("single_hop", "multi_hop", "chained_multi_hop")
        for index in range(2)
    }
    records = [
        _record_for(item_id=item_id, system_id=system_id)
        for system_id in ("flat_vector", "map_reduce")
        for item_id in items
    ]
    write_foundry_evaluation_export(
        store=store,
        records=records,
        items_by_id=items,
        config=foundry_config,
        experiment_id=config.experiment.id,
    )
    calls: list[tuple[str, list[dict[str, Any]]]] = []

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        rows = [
            json.loads(line)
            for line in Path(kwargs["data"]).read_text(encoding="utf-8").splitlines()
        ]
        system_id = rows[0]["system_id"]
        calls.append((system_id, rows))
        score = 2.0 if system_id == "flat_vector" else 4.0
        return {
            "metrics": {"groundedness.score": score},
            "rows": rows,
        }

    result = run_foundry_managed_evaluation(
        config=config,
        store=store,
        evaluate_fn=fake_evaluate,
        sleep_fn=lambda _: None,
    )

    assert [system_id for system_id, _ in calls] == ["flat_vector", "map_reduce"]
    assert all(len(rows) == 3 for _, rows in calls)
    assert {row["item_id"] for row in calls[0][1]} == {row["item_id"] for row in calls[1][1]}
    assert result.row_count == 6
    assert result.metrics == {"groundedness.score": 3.0}
    assert result.metadata["selected_item_count"] == 3
    assert result.metadata["source_row_count"] == 12
    assert set(result.metadata["scope_results"]) == {"flat_vector", "map_reduce"}
    assert result.metadata["scope_results"]["flat_vector"]["metrics"] == {
        "groundedness.score": 2.0
    }
    assert (
        store.experiment_dir
        / "evaluations/foundry/managed-evaluators/flat_vector/groundedness.json"
    ).exists()


def test_foundry_managed_evaluator_config_maps_rag_fields() -> None:
    evaluator_config = _managed_evaluator_config(
        [
            "groundedness",
            "relevance",
            "retrieval",
            "document_retrieval",
            "response_completeness",
            "qa",
            "similarity",
        ]
    )

    assert evaluator_config["relevance"]["column_mapping"] == {
        "query": "${data.query}",
        "response": "${data.response}",
    }
    assert evaluator_config["retrieval"]["column_mapping"] == {
        "query": "${data.query}",
        "context": "${data.context}",
    }
    assert evaluator_config["response_completeness"]["column_mapping"] == {
        "response": "${data.response}",
        "ground_truth": "${data.ground_truth}",
    }
    assert evaluator_config["qa"]["column_mapping"] == {
        "query": "${data.query}",
        "response": "${data.response}",
        "context": "${data.context}",
        "ground_truth": "${data.ground_truth}",
    }
    assert evaluator_config["similarity"]["column_mapping"] == {
        "response": "${data.response}",
        "ground_truth": "${data.ground_truth}",
    }


def test_foundry_managed_judge_endpoint_is_derived_from_generator_base_url() -> None:
    assert (
        _resolved_azure_openai_endpoint(
            "https://example.services.ai.azure.com/openai/v1/responses"
        )
        == "https://example.services.ai.azure.com"
    )
    assert (
        _resolved_azure_openai_endpoint(
            "https://example.openai.azure.com/openai/deployments/gpt-5/responses"
        )
        == "https://example.openai.azure.com"
    )


def test_foundry_managed_evaluators_mark_gpt5_judge_as_reasoning_model(
    monkeypatch,
) -> None:
    calls: dict[str, dict[str, Any]] = {}

    fake_module = _fake_azure_evaluation_module(calls)
    monkeypatch.setitem(sys.modules, "azure.ai.evaluation", fake_module)
    monkeypatch.setenv("AZURE_INFERENCE_CREDENTIAL", "test-key")
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"id": "exp"},
            "dataset": {"adapter": "jsonl"},
            "models": {
                "generator_provider": "foundry",
                "generator_base_url": "https://example.services.ai.azure.com/openai/v1/",
                "generator_deployment": "gpt-5-mini-doc-map-generator",
                "judge_deployment": "gpt-5-rag-judge",
            },
            "systems": ["flat_vector"],
        }
    )

    _managed_evaluators(
        [
            "groundedness",
            "relevance",
            "retrieval",
            "response_completeness",
            "document_retrieval",
        ],
        config=config,
    )

    assert calls["GroundednessEvaluator"]["kwargs"]["is_reasoning_model"] is True
    assert calls["RelevanceEvaluator"]["kwargs"]["is_reasoning_model"] is True
    assert calls["RetrievalEvaluator"]["kwargs"]["is_reasoning_model"] is True
    assert calls["ResponseCompletenessEvaluator"]["kwargs"]["is_reasoning_model"] is True
    assert calls["DocumentRetrievalEvaluator"]["kwargs"] == {}


def test_foundry_managed_evaluators_allow_reasoning_model_override(
    monkeypatch,
) -> None:
    calls: dict[str, dict[str, Any]] = {}

    fake_module = _fake_azure_evaluation_module(calls)
    monkeypatch.setitem(sys.modules, "azure.ai.evaluation", fake_module)
    monkeypatch.setenv("AZURE_INFERENCE_CREDENTIAL", "test-key")
    monkeypatch.setenv("FOUNDRY_EVALUATION_REASONING_MODEL", "false")
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"id": "exp"},
            "dataset": {"adapter": "jsonl"},
            "models": {
                "generator_provider": "foundry",
                "generator_base_url": "https://example.services.ai.azure.com/openai/v1/",
                "generator_deployment": "gpt-5-mini-doc-map-generator",
                "judge_deployment": "gpt-5-rag-judge",
            },
            "systems": ["flat_vector"],
        }
    )

    _managed_evaluators(["groundedness"], config=config)

    assert calls["GroundednessEvaluator"]["kwargs"]["is_reasoning_model"] is False


def test_foundry_managed_evaluators_require_explicit_judge_deployment(
    monkeypatch,
) -> None:
    calls: dict[str, dict[str, Any]] = {}
    monkeypatch.setitem(sys.modules, "azure.ai.evaluation", _fake_azure_evaluation_module(calls))
    monkeypatch.setenv("AZURE_INFERENCE_CREDENTIAL", "test-key")
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"id": "exp"},
            "dataset": {"adapter": "jsonl"},
            "models": {
                "generator_provider": "foundry",
                "generator_base_url": "https://example.services.ai.azure.com/openai/v1/",
                "generator_deployment": "gpt-5-mini-doc-map-generator",
            },
            "systems": ["flat_vector"],
        }
    )

    with pytest.raises(RuntimeError, match="models.judge_deployment"):
        _managed_evaluators(["groundedness"], config=config)


def test_reasoning_model_deployment_detection() -> None:
    assert _is_reasoning_model_deployment("gpt-5-mini-doc-map-generator") is True
    assert _is_reasoning_model_deployment("judge-o3") is True
    assert _is_reasoning_model_deployment("prod-gpt-4o-mini") is False
    assert _is_reasoning_model_deployment("prod-judge") is False


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


def _fake_azure_evaluation_module(calls: dict[str, dict[str, Any]]) -> ModuleType:
    module = ModuleType("azure.ai.evaluation")

    def evaluator_class(name: str):
        class FakeEvaluator:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                calls[name] = {"args": args, "kwargs": kwargs}

        FakeEvaluator.__name__ = name
        return FakeEvaluator

    for name in (
        "BleuScoreEvaluator",
        "DocumentRetrievalEvaluator",
        "F1ScoreEvaluator",
        "GleuScoreEvaluator",
        "GroundednessEvaluator",
        "MeteorScoreEvaluator",
        "QAEvaluator",
        "RelevanceEvaluator",
        "ResponseCompletenessEvaluator",
        "RetrievalEvaluator",
        "RougeScoreEvaluator",
        "SimilarityEvaluator",
    ):
        setattr(module, name, evaluator_class(name))

    module.RougeType = type("RougeType", (), {"ROUGE_1": "rouge_1"})
    return module


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


def _record_for(*, item_id: str, system_id: str) -> RagRunRecord:
    record = _record()
    return record.model_copy(
        update={
            "run_id": f"run-{system_id}-{item_id}",
            "system_id": system_id,
            "item_id": item_id,
        }
    )
