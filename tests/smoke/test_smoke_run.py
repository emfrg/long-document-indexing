from __future__ import annotations

import asyncio
from pathlib import Path

from long_document_indexing.cli import _evaluate, _index, _prepare, _query, _report
from long_document_indexing.config import load_experiment_config


def test_smoke_run_writes_metrics_and_report(tmp_path) -> None:
    config = load_experiment_config(
        Path("configs/experiments/smoke-test.yaml"),
        project_root=Path.cwd(),
    )
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(update={"artifacts_dir": tmp_path / "artifacts"})
        }
    )

    _prepare(config)
    asyncio.run(_index(config))
    asyncio.run(_query(config))
    _evaluate(config)
    _report(config)

    experiment_dir = tmp_path / "artifacts" / "smoke-test"
    assert (experiment_dir / "evaluations" / "local-metrics.jsonl").exists()
    assert (experiment_dir / "report" / "results.md").exists()
    assert (experiment_dir / "report" / "results.csv").exists()
    assert (experiment_dir / "report" / "system-summary.csv").exists()
    assert (experiment_dir / "report" / "usage-summary.csv").exists()
    assert (experiment_dir / "report" / "summary.json").exists()


def test_foundry_eval_export_smoke_writes_dataset_and_manifest(tmp_path) -> None:
    config = load_experiment_config(
        Path("configs/experiments/foundry-eval-export-smoke.yaml"),
        project_root=Path.cwd(),
    )
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(update={"artifacts_dir": tmp_path / "artifacts"})
        }
    )

    _prepare(config)
    asyncio.run(_index(config))
    asyncio.run(_query(config))
    _evaluate(config)
    _report(config)

    experiment_dir = tmp_path / "artifacts" / "foundry-eval-export-smoke"
    dataset_path = experiment_dir / "evaluations" / "foundry" / "dataset.jsonl"
    manifest_path = experiment_dir / "evaluations" / "foundry" / "manifest.json"
    assert dataset_path.exists()
    assert manifest_path.exists()
    assert len(dataset_path.read_text(encoding="utf-8").splitlines()) == 2
    assert (experiment_dir / "report" / "system-summary.csv").exists()
    assert (experiment_dir / "report" / "usage-summary.csv").exists()
    assert (experiment_dir / "report" / "summary.json").exists()


def test_enterprise_thin_slice_smoke_writes_report_and_foundry_export(tmp_path) -> None:
    config = load_experiment_config(
        Path("configs/experiments/enterprise-thin-slice.yaml"),
        project_root=Path.cwd(),
    )
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(update={"artifacts_dir": tmp_path / "artifacts"})
        }
    )

    _prepare(config)
    asyncio.run(_index(config))
    asyncio.run(_query(config))
    _evaluate(config)
    _report(config)

    experiment_dir = tmp_path / "artifacts" / "enterprise-thin-slice"
    dataset_path = experiment_dir / "evaluations" / "foundry" / "dataset.jsonl"
    manifest_path = experiment_dir / "evaluations" / "foundry" / "manifest.json"
    assert (experiment_dir / "evaluations" / "local-metrics.jsonl").exists()
    assert dataset_path.exists()
    assert manifest_path.exists()
    assert len(dataset_path.read_text(encoding="utf-8").splitlines()) == 64
    assert (experiment_dir / "report" / "results.md").exists()
    assert (experiment_dir / "report" / "system-summary.csv").exists()
    assert (experiment_dir / "report" / "usage-summary.csv").exists()
    assert (experiment_dir / "report" / "summary.json").exists()


def test_advanced_systems_smoke_writes_report_and_foundry_export(tmp_path) -> None:
    config = load_experiment_config(
        Path("configs/experiments/advanced-systems-smoke.yaml"),
        project_root=Path.cwd(),
    )
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(update={"artifacts_dir": tmp_path / "artifacts"})
        }
    )

    _prepare(config)
    asyncio.run(_index(config))
    asyncio.run(_query(config))
    _evaluate(config)
    _report(config)

    experiment_dir = tmp_path / "artifacts" / "advanced-systems-smoke"
    dataset_path = experiment_dir / "evaluations" / "foundry" / "dataset.jsonl"
    assert (experiment_dir / "evaluations" / "local-metrics.jsonl").exists()
    assert dataset_path.exists()
    assert len(dataset_path.read_text(encoding="utf-8").splitlines()) == 14
    assert (experiment_dir / "report" / "results.md").exists()
    assert (experiment_dir / "report" / "system-summary.csv").exists()
    assert (experiment_dir / "report" / "summary.json").exists()
