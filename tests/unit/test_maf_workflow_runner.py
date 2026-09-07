from __future__ import annotations

import importlib.util

import pytest

from long_document_indexing.domain.runs import RunContext
from long_document_indexing.workflows.execution import MafWorkflowRunner

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("agent_framework") is None,
    reason="agent_framework is not installed",
)


async def test_maf_workflow_runner_executes_operation_and_records_framework_events() -> None:
    runner = MafWorkflowRunner()
    context = RunContext(
        experiment_id="exp",
        run_id="run",
        system_id="stuffing",
        corpus_id="corpus",
    )

    async def operation() -> str:
        return "done"

    result = await runner.run("unit_maf", context, operation)

    assert result.output == "done"
    assert result.record.status == "succeeded"
    assert result.record.metadata["runner"] == "maf"
    assert result.record.metadata["final_state"] == "IDLE"
    assert result.record.trace_id.startswith("trace-")
    assert any(event.event_type == "executor_completed" for event in result.record.events)
    assert all(event.metadata.get("runner") != "local" for event in result.record.events)
