from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from long_document_indexing.cli import _index, _prepare, _query
from long_document_indexing.config import RunControlConfig, load_experiment_config
from long_document_indexing.domain.runs import RagRunRecord
from long_document_indexing.run_control import BudgetExceeded
from long_document_indexing.telemetry.usage import UsageEvent


def test_resume_reuses_succeeded_index_and_query_artifacts(tmp_path) -> None:
    config = _foundry_export_smoke_config(tmp_path)

    _prepare(config)
    asyncio.run(_index(config))
    asyncio.run(_query(config))

    experiment_dir = tmp_path / "artifacts" / "foundry-eval-export-smoke"
    index_usage_count = _jsonl_count(experiment_dir / "costs/index-usage.jsonl")
    query_usage_count = _jsonl_count(experiment_dir / "costs/query-usage.jsonl")
    run_count = _jsonl_count(experiment_dir / "runs/stuffing.jsonl")

    resume_config = config.model_copy(update={"run_control": RunControlConfig(resume=True)})
    asyncio.run(_index(resume_config))
    asyncio.run(_query(resume_config))

    records = _run_records(experiment_dir / "runs/stuffing.jsonl")
    assert _jsonl_count(experiment_dir / "costs/index-usage.jsonl") == index_usage_count
    assert _jsonl_count(experiment_dir / "costs/query-usage.jsonl") == query_usage_count
    assert _jsonl_count(experiment_dir / "runs/stuffing.jsonl") == run_count
    assert len(records) == 2
    assert {record.status for record in records} == {"succeeded"}


def test_exhausted_query_budget_writes_skipped_records(tmp_path) -> None:
    config = _foundry_export_smoke_config(tmp_path)
    budget_config = config.model_copy(update={"run_control": RunControlConfig(max_model_calls=0)})

    _prepare(config)
    asyncio.run(_index(config))
    asyncio.run(_query(budget_config))

    experiment_dir = tmp_path / "artifacts" / "foundry-eval-export-smoke"
    records = _run_records(experiment_dir / "runs/stuffing.jsonl")
    query_usage = [
        UsageEvent.model_validate_json(line)
        for line in (experiment_dir / "costs/query-usage.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert len(records) == 2
    assert {record.status for record in records} == {"skipped"}
    assert all("Budget exhausted before query stuffing" in str(record.error) for record in records)
    assert query_usage == []


def test_index_budget_persists_completed_artifact_before_raising(tmp_path) -> None:
    config = _foundry_export_smoke_config(tmp_path).model_copy(
        update={"run_control": RunControlConfig(max_model_calls=1)}
    )

    _prepare(config)
    with pytest.raises(BudgetExceeded, match="Budget exceeded after index stuffing"):
        asyncio.run(_index(config))

    experiment_dir = tmp_path / "artifacts" / "foundry-eval-export-smoke"
    assert _jsonl_count(experiment_dir / "indexes/index_artifacts.jsonl") == 1
    assert _jsonl_count(experiment_dir / "costs/index-usage.jsonl") == 2


def _foundry_export_smoke_config(tmp_path: Path):
    config = load_experiment_config(
        Path("configs/experiments/foundry-eval-export-smoke.yaml"),
        project_root=Path.cwd(),
    )
    return config.model_copy(
        update={
            "storage": config.storage.model_copy(update={"artifacts_dir": tmp_path / "artifacts"})
        }
    )


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def _run_records(path: Path) -> list[RagRunRecord]:
    return [
        RagRunRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
