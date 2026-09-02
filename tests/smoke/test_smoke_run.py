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
