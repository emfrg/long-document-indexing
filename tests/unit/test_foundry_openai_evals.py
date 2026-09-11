from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from long_document_indexing.config import load_experiment_config
from long_document_indexing.evaluation.foundry.openai_evals import (
    OPENAI_EVALS_SYSTEM_RESULTS_PATH,
    build_foundry_openai_evals_plan,
    run_foundry_openai_evals_for_system,
    run_foundry_openai_evals_per_system,
)
from long_document_indexing.storage.artifacts import ArtifactStore


def test_foundry_openai_evals_plan_uses_env_project(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(
        "FOUNDRY_EVALUATION_PROJECT_ENDPOINT",
        "https://example.services.ai.azure.com/api/projects/project-a",
    )
    config = _config(tmp_path)
    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)
    _write_dataset(store, config)

    plan = build_foundry_openai_evals_plan(
        config=config,
        store=store,
        evaluation_name="portal-eval",
        run_name="portal-run",
    )

    assert plan["evaluation_name"] == "portal-eval"
    assert plan["mode"] == "system_comparison"
    assert plan["runs"] == [
        {
            "system_id": "stuffing",
            "run_name": "portal-run-stuffing",
            "run_dataset_path": str(
                store.experiment_dir
                / "evaluations/foundry/openai-evals-dataset-stuffing.jsonl"
            ),
            "result_path": str(
                store.experiment_dir
                / "evaluations/foundry/openai-evals-result-stuffing.json"
            ),
        }
    ]
    assert plan["dataset_exists"] is True
    assert plan["project_endpoint"] == "https://example.services.ai.azure.com/api/projects/project-a"
    assert [criterion["name"] for criterion in plan["testing_criteria"]] == [
        "token_f1",
        "rouge_1",
    ]


def test_foundry_openai_evals_plan_supports_rag_criteria(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(
        "FOUNDRY_EVALUATION_PROJECT_ENDPOINT",
        "https://example.services.ai.azure.com/api/projects/project-a",
    )
    config = load_experiment_config(
        Path("configs/experiments/foundry-multilexsum-legal-rag-qa-smoke.yaml"),
        project_root=Path.cwd(),
    )
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(update={"artifacts_dir": tmp_path / "artifacts"})
        }
    )
    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)
    _write_rag_dataset(store, config)

    plan = build_foundry_openai_evals_plan(
        config=config,
        store=store,
        evaluation_name="portal-rag-eval",
        run_name="portal-rag-run",
    )

    assert plan["dataset_exists"] is True
    assert [criterion["name"] for criterion in plan["testing_criteria"]] == [
        "citation_support_rate",
        "answer_reference_token_f1",
        "context_precision_at_4",
        "context_recall_at_4",
        "evidence_quote_recall_at_4",
        "document_retrieval_precision",
        "document_retrieval_recall",
        "answer_reference_token_recall",
    ]
    assert {criterion["type"] for criterion in plan["testing_criteria"]} == {"python"}


def test_foundry_openai_evals_for_system_uploads_file_id_run_and_writes_result(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "FOUNDRY_EVALUATION_PROJECT_ENDPOINT",
        "https://example.services.ai.azure.com/api/projects/project-a",
    )
    config = _config(tmp_path)
    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)
    row = _write_dataset(store, config)
    client = FakeOpenAIEvalsClient()
    statuses: list[str] = []

    result = run_foundry_openai_evals_for_system(
        config=config,
        store=store,
        system_id="stuffing",
        evaluation_name="portal-eval",
        run_name="portal-run",
        client=client,
        sleep_fn=lambda _: None,
        status_callback=statuses.append,
    )

    assert client.files.created_purpose == "evals"
    assert client.evals.created["name"] == "portal-eval"
    assert (
        "retrieved_context"
        in client.evals.created["data_source_config"]["item_schema"]["properties"]
    )
    assert client.evals.runs.created["eval_id"] == "eval-created"
    assert client.evals.runs.created["data_source"]["source"] == {
        "type": "file_id",
        "id": "file-created",
    }
    assert result.eval_id == "eval-created"
    assert result.run_id == "run-created"
    assert result.system_id == "stuffing"
    assert result.report_url == "https://ai.azure.com/report"
    assert result.score_averages == {"rouge_1": 0.25, "token_f1": 0.75}
    assert (
        store.experiment_dir / "evaluations/foundry/openai-evals-result-stuffing.json"
    ).exists()

    run_dataset_row = json.loads(
        (store.experiment_dir / "evaluations/foundry/openai-evals-dataset-stuffing.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert run_dataset_row == {"item": row}
    assert statuses == [
        "Preparing evaluation: portal-eval",
        "Preparing upload dataset: evaluations/foundry/openai-evals-dataset-stuffing.jsonl",
        "Uploading dataset and waiting for file processing",
        "Waiting for uploaded file to process: file-created",
        "Creating run: portal-run",
        "Waiting for run to finish: run-created",
        "Run status: completed",
        "Fetching output items",
        "Writing result: evaluations/foundry/openai-evals-result-stuffing.json",
    ]


def test_foundry_openai_evals_per_system_creates_run_per_system(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "FOUNDRY_EVALUATION_PROJECT_ENDPOINT",
        "https://example.services.ai.azure.com/api/projects/project-a",
    )
    config = _config(tmp_path)
    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)
    rows = [
        {
            "id": "run-stuffing",
            "query": "What did the board approve?",
            "response": "The board approved the Alpha renewal.",
            "ground_truth": "The board approved the Alpha renewal.",
            "system_id": "stuffing",
            "item_id": "q_alpha",
        },
        {
            "id": "run-map-reduce",
            "query": "What did the board approve?",
            "response": "The board approved the Alpha renewal.",
            "ground_truth": "The board approved the Alpha renewal.",
            "system_id": "map_reduce",
            "item_id": "q_alpha",
        },
    ]
    store.write_jsonl(config.evaluation.foundry.dataset_path, rows)
    client = FakeOpenAIEvalsClient()

    result = run_foundry_openai_evals_per_system(
        config=config,
        store=store,
        evaluation_name="system-comparison",
        run_name="comparison",
        client=client,
        sleep_fn=lambda _: None,
    )

    assert result.mode == "system_comparison"
    assert result.systems == ["stuffing", "map_reduce"]
    assert [run.run_name for run in result.results] == [
        "comparison-stuffing",
        "comparison-map_reduce",
    ]
    assert [run.system_id for run in result.results] == ["stuffing", "map_reduce"]
    assert [run["name"] for run in client.evals.runs.created_runs] == [
        "comparison-stuffing",
        "comparison-map_reduce",
    ]
    assert (
        store.experiment_dir / OPENAI_EVALS_SYSTEM_RESULTS_PATH
    ).exists()

    stuffing_rows = (
        store.experiment_dir / "evaluations/foundry/openai-evals-dataset-stuffing.jsonl"
    ).read_text(encoding="utf-8")
    map_reduce_rows = (
        store.experiment_dir / "evaluations/foundry/openai-evals-dataset-map_reduce.jsonl"
    ).read_text(encoding="utf-8")
    assert '"system_id": "stuffing"' in stuffing_rows
    assert '"system_id": "map_reduce"' not in stuffing_rows
    assert '"system_id": "map_reduce"' in map_reduce_rows
    assert '"system_id": "stuffing"' not in map_reduce_rows


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


def _write_dataset(store: ArtifactStore, config) -> dict[str, Any]:
    row = {
        "id": "run-alpha",
        "query": "What did the board approve?",
        "response": "The board approved the Alpha renewal.",
        "ground_truth": "The board approved the Alpha renewal.",
        "system_id": "stuffing",
        "item_id": "q_alpha",
    }
    store.write_jsonl(config.evaluation.foundry.dataset_path, [row])
    return row


def _write_rag_dataset(store: ArtifactStore, config) -> dict[str, Any]:
    row = {
        "id": "run-alpha",
        "query": "What did the court require?",
        "response": "The court required an ability-to-pay inquiry.",
        "context": "[rank=1 document_id=doc segment_id=seg]\nability-to-pay inquiry",
        "ground_truth": "The court required an ability-to-pay inquiry.",
        "messages": [],
        "retrieved_context": [
            {
                "document_id": "doc",
                "segment_id": "seg",
                "rank": 1,
                "text": "ability-to-pay inquiry",
                "retrieval_stage": "test",
            }
        ],
        "retrieved_documents": [{"document_id": "doc", "rank": 1}],
        "retrieval_ground_truth": [{"document_id": "doc", "query_relevance_label": 4}],
        "citations": [
            {
                "document_id": "doc",
                "segment_id": "seg",
                "quote": "ability-to-pay inquiry",
            }
        ],
        "expected_behavior": {
            "expected_answer": "The court required an ability-to-pay inquiry.",
            "relevant_document_ids": ["doc"],
            "relevant_segment_ids": ["seg"],
            "evidence": [
                {
                    "document_id": "doc",
                    "segment_id": "seg",
                    "quote": "ability-to-pay inquiry",
                }
            ],
        },
        "experiment_id": "exp",
        "run_id": "run-alpha",
        "system_id": "stuffing",
        "corpus_id": "corpus",
        "item_id": "q_alpha",
        "repetition": 1,
        "selected_document_ids": ["doc"],
        "relevant_document_ids": ["doc"],
        "relevant_segment_ids": ["seg"],
        "tags": ["legal_rag_qa"],
        "status": "succeeded",
        "metadata": {},
    }
    store.write_jsonl(config.evaluation.foundry.dataset_path, [row])
    return row


class FakeOpenAIEvalsClient:
    def __init__(self) -> None:
        self.files = FakeFiles()
        self.evals = FakeEvals()


class FakeFiles:
    def __init__(self) -> None:
        self.created_purpose: str | None = None

    def create(self, *, file, purpose: str, timeout: int) -> SimpleNamespace:
        del timeout
        self.created_purpose = purpose
        file.read()
        return SimpleNamespace(id="file-created")

    def wait_for_processing(
        self,
        file_id: str,
        *,
        poll_interval: int,
        max_wait_seconds: int,
    ) -> SimpleNamespace:
        del poll_interval, max_wait_seconds
        return SimpleNamespace(id=file_id, status="processed")


class FakeEvals:
    def __init__(self) -> None:
        self.created: dict[str, Any] = {}
        self.runs = FakeRuns()

    def list(self, **kwargs: Any) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(data=[], has_more=False)

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.created = kwargs
        return SimpleNamespace(id="eval-created", name=kwargs["name"])


class FakeRuns:
    def __init__(self) -> None:
        self.created: dict[str, Any] = {}
        self.created_runs: list[dict[str, Any]] = []
        self.output_items = FakeOutputItems()

    def create(self, eval_id: str, **kwargs: Any) -> SimpleNamespace:
        self.created = {"eval_id": eval_id, **kwargs}
        self.created_runs.append(self.created)
        return SimpleNamespace(id="run-created", status="queued")

    def retrieve(self, run_id: str, *, eval_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=run_id,
            eval_id=eval_id,
            status="completed",
            report_url="https://ai.azure.com/report",
            result_counts=SimpleNamespace(total=1, passed=1, failed=0, errored=0),
        )


class FakeOutputItems:
    def list(self, run_id: str, **kwargs: Any) -> SimpleNamespace:
        del run_id, kwargs
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    results=[
                        SimpleNamespace(name="token_f1", score=0.75),
                        SimpleNamespace(name="rouge_1", score=0.25),
                    ]
                )
            ],
            has_more=False,
        )
