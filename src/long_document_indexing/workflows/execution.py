from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar, cast

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
    metadata: dict[str, Any] = Field(default_factory=dict)


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
                {"trace_id": trace_id, "runner": "local"},
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
                    {"error_type": exc.__class__.__name__, "runner": "local"},
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
                metadata={"runner": "local"},
            )
            raise WorkflowExecutionError(record) from exc

        duration_ms = (time.perf_counter() - started) * 1000.0
        events.append(
            _event(
                workflow_name,
                "workflow_completed",
                "workflow",
                "Workflow completed.",
                {"trace_id": trace_id, "runner": "local"},
            )
        )
        record = WorkflowRunRecord(
            workflow_name=workflow_name,
            run_context=context,
            trace_id=trace_id,
            status="succeeded",
            duration_ms=duration_ms,
            events=events,
            metadata={"runner": "local"},
        )
        return WorkflowResult(output=output, record=record)


class MafWorkflowRunner:
    """Workflow runner backed by the Microsoft Agent Framework functional API."""

    def __init__(self, *, include_status_events: bool = True) -> None:
        try:
            import agent_framework  # noqa: F401
        except ImportError as exc:
            raise WorkflowIntegrationNotAvailable(
                "Microsoft Agent Framework is not installed. Install the optional 'maf' extra "
                "before selecting the MAF workflow runner."
            ) from exc
        self.include_status_events = include_status_events

    async def run(
        self,
        workflow_name: str,
        context: RunContext,
        operation: Callable[[], Awaitable[OutputT]],
    ) -> WorkflowResult[OutputT]:
        try:
            from agent_framework import step, workflow
        except ImportError as exc:
            raise WorkflowIntegrationNotAvailable(
                "Microsoft Agent Framework is not installed. Install the optional 'maf' extra "
                "before selecting the MAF workflow runner."
            ) from exc

        trace_id = stable_trace_id(context, workflow_name)
        started = time.perf_counter()
        events = [
            _event(
                workflow_name,
                "workflow_started",
                "workflow",
                "Workflow started.",
                {
                    "trace_id": trace_id,
                    "runner": "maf",
                    "framework": "microsoft_agent_framework",
                },
            )
        ]

        @step(name=f"{workflow_name}_operation")
        async def execute_operation(_: dict[str, Any]) -> OutputT:
            return await operation()

        @workflow(name=workflow_name, description=f"{workflow_name} workflow")
        async def execute_workflow(payload: dict[str, Any]) -> OutputT:
            return await execute_operation(payload)

        try:
            maf_result = await execute_workflow.build().run(
                context.model_dump(mode="json"),
                include_status_events=self.include_status_events,
            )
            events.extend(_maf_event_records(workflow_name, maf_result))
            final_state = _enum_name(maf_result.get_final_state())
            outputs = maf_result.get_outputs()
            if final_state != "IDLE":
                raise _MafWorkflowStateError(
                    f"MAF workflow ended in unsupported state {final_state}"
                )
            if not outputs:
                raise _MafWorkflowStateError("MAF workflow completed without output")
            output = cast(OutputT, outputs[-1])
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            events.append(
                _event(
                    workflow_name,
                    "workflow_failed",
                    "workflow",
                    "Workflow failed.",
                    {
                        "runner": "maf",
                        "framework": "microsoft_agent_framework",
                        "error_type": exc.__class__.__name__,
                    },
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
                metadata={
                    "runner": "maf",
                    "framework": "microsoft_agent_framework",
                },
            )
            raise WorkflowExecutionError(record) from exc

        duration_ms = (time.perf_counter() - started) * 1000.0
        events.append(
            _event(
                workflow_name,
                "workflow_completed",
                "workflow",
                "Workflow completed.",
                {
                    "trace_id": trace_id,
                    "runner": "maf",
                    "framework": "microsoft_agent_framework",
                    "final_state": final_state,
                },
            )
        )
        record = WorkflowRunRecord(
            workflow_name=workflow_name,
            run_context=context,
            trace_id=trace_id,
            status="succeeded",
            duration_ms=duration_ms,
            events=events,
            metadata={
                "runner": "maf",
                "framework": "microsoft_agent_framework",
                "final_state": final_state,
            },
        )
        return WorkflowResult(output=output, record=record)


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


class _MafWorkflowStateError(RuntimeError):
    """Raised when MAF execution completes in a state this benchmark runner cannot use."""


def _maf_event_records(workflow_name: str, maf_result: Any) -> list[WorkflowEventRecord]:
    return [_maf_event_record(workflow_name, event) for event in maf_result]


def _maf_event_record(workflow_name: str, event: Any) -> WorkflowEventRecord:
    event_type = str(_get(event, "type", "maf_event"))
    stage = str(_get(event, "executor_id") or "workflow")
    metadata: dict[str, Any] = {
        "runner": "maf",
        "framework": "microsoft_agent_framework",
        "maf_event_type": event_type,
    }
    state = _get(event, "state")
    if state is not None:
        metadata["state"] = _enum_name(state)
    data = _get(event, "data")
    if data is not None:
        metadata["data_type"] = data.__class__.__name__
    return _event(
        workflow_name,
        event_type,
        stage,
        _maf_event_message(event_type, stage),
        metadata,
    )


def _maf_event_message(event_type: str, stage: str) -> str:
    return f"MAF event {event_type} from {stage}."


def _enum_name(value: Any) -> str:
    name = _get(value, "name")
    if isinstance(name, str):
        return name
    return str(value)


def _get(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
