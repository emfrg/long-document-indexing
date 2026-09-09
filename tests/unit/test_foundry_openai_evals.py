from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from long_document_indexing.config import load_experiment_config
from long_document_indexing.evaluation.foundry.openai_evals import (
    OPENAI_EVALS_DATASET_PATH,
    build_foundry_openai_evals_plan,
    run_foundry_openai_evals,
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
    assert plan["run_name"] == "portal-run"
    assert plan["dataset_exists"] is True
    assert plan["project_endpoint"] == "https://example.services.ai.azure.com/api/projects/project-a"
    assert [criterion["name"] for criterion in plan["testing_criteria"]] == [
        "token_f1",
        "rouge_1",
    ]


def test_foundry_openai_evals_uploads_file_id_run_and_writes_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(
        "FOUNDRY_EVALUATION_PROJECT_ENDPOINT",
        "https://example.services.ai.azure.com/api/projects/project-a",
    )
    config = _config(tmp_path)
    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)
    row = _write_dataset(store, config)
    client = FakeOpenAIEvalsClient()

    result = run_foundry_openai_evals(
        config=config,
        store=store,
        evaluation_name="portal-eval",
        run_name="portal-run",
        client=client,
        sleep_fn=lambda _: None,
    )

    assert client.files.created_purpose == "evals"
    assert client.evals.created["name"] == "portal-eval"
    assert client.evals.runs.created["eval_id"] == "eval-created"
    assert client.evals.runs.created["data_source"]["source"] == {
        "type": "file_id",
        "id": "file-created",
    }
    assert result.eval_id == "eval-created"
    assert result.run_id == "run-created"
    assert result.report_url == "https://ai.azure.com/report"
    assert result.score_averages == {"rouge_1": 0.25, "token_f1": 0.75}
    assert (store.experiment_dir / "evaluations/foundry/openai-evals-result.json").exists()

    run_dataset_row = json.loads(
        (store.experiment_dir / OPENAI_EVALS_DATASET_PATH)
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert run_dataset_row == {"item": row}


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
        self.output_items = FakeOutputItems()

    def create(self, eval_id: str, **kwargs: Any) -> SimpleNamespace:
        self.created = {"eval_id": eval_id, **kwargs}
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
