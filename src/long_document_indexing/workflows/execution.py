from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from long_document_indexing.domain.runs import RunContext
from long_document_indexing.telemetry.tracing import stable_trace_id

OutputT = TypeVar("OutputT")


class WorkflowIntegrationNotAvailable(RuntimeError):
    """Raised when a later milestone tries to use workflow integration before it exists."""


class WorkflowEventRecord(BaseModel):
    """One local event emitted while a workflow executes."""

    model_config = ConfigDict(extra="forbid")

    workflow_name: str
    event_type: str
    stage: str
    message: str
    timestamp: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowRunRecord(BaseModel):
    """Serializable execution envelope for a workflow run."""

    model_config = ConfigDict(extra="forbid")

    workflow_name: str
    run_context: RunContext
    trace_id: str
    status: str
    duration_ms: float
    events: list[WorkflowEventRecord]
    error: str | None = None


@dataclass(frozen=True)
class WorkflowResult[OutputT]:
    """Workflow execution result plus the strongly typed application output."""

    output: OutputT
    record: WorkflowRunRecord


class WorkflowRunner(Protocol):
    """Runtime boundary for executing workflows."""

    async def run(
        self,
        workflow_name: str,
        context: RunContext,
        operation: Callable[[], Awaitable[OutputT]],
    ) -> WorkflowResult[OutputT]:
        """Execute an async operation as a named workflow."""


class LocalWorkflowRunner:
    """Workflow runner that records local events around plain Python async functions."""

    async def run(
        self,
        workflow_name: str,
        context: RunContext,
        operation: Callable[[], Awaitable[OutputT]],
    ) -> WorkflowResult[OutputT]:
        trace_id = stable_trace_id(context, workflow_name)
        started = time.perf_counter()
        events = [
            _event(
                workflow_name,
                "workflow_started",
                "workflow",
                "Workflow started.",
                {"trace_id": trace_id},
            )
        ]

        try:
            output = await operation()
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            events.append(
                _event(
                    workflow_name,
                    "workflow_failed",
                    "workflow",
                    "Workflow failed.",
                    {"error_type": exc.__class__.__name__},
                )
            )
            record = WorkflowRunRecord(
                workflow_name=workflow_name,
                run_context=context,
                trace_id=trace_id,
                status="failed",
                duration_ms=duration_ms,
                events=events,
                error=str(exc),
            )
            raise WorkflowExecutionError(record) from exc

        duration_ms = (time.perf_counter() - started) * 1000.0
        events.append(
            _event(
                workflow_name,
                "workflow_completed",
                "workflow",
                "Workflow completed.",
                {"trace_id": trace_id},
            )
        )
        record = WorkflowRunRecord(
            workflow_name=workflow_name,
            run_context=context,
            trace_id=trace_id,
            status="succeeded",
            duration_ms=duration_ms,
            events=events,
        )
        return WorkflowResult(output=output, record=record)


class MafWorkflowRunner:
    """Placeholder runner for a later real Microsoft Agent Framework adapter."""

    def __init__(self) -> None:
        try:
            import agent_framework  # noqa: F401
        except ImportError as exc:
            raise WorkflowIntegrationNotAvailable(
                "Microsoft Agent Framework is not installed. Install the optional 'maf' extra "
                "before selecting the MAF workflow runner."
            ) from exc

        raise WorkflowIntegrationNotAvailable(
            "The MAF workflow runner boundary exists, but concrete execution is scheduled for "
            "the first model-backed workflow milestone."
        )


class WorkflowExecutionError(RuntimeError):
    """Wraps a failed workflow record while preserving the original exception as cause."""

    def __init__(self, record: WorkflowRunRecord) -> None:
        super().__init__(record.error or "workflow execution failed")
        self.record = record


def _event(
    workflow_name: str,
    event_type: str,
    stage: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> WorkflowEventRecord:
    return WorkflowEventRecord(
        workflow_name=workflow_name,
        event_type=event_type,
        stage=stage,
        message=message,
        timestamp=datetime.now(UTC).isoformat(),
        metadata=metadata or {},
    )
