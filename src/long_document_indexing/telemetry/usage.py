from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from long_document_indexing.domain.runs import RunContext, UsageRecord


class UsageEvent(BaseModel):
    """Atomic usage or timing event attached to one benchmark run."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    run_id: str
    system_id: str
    corpus_id: str
    item_id: str | None = None
    repetition: int = 0
    stage: str
    kind: Literal["workflow", "model", "tool", "system"]
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    duration_ms: float = 0.0
    estimated_cost: float | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


class UsageLedger:
    """In-memory usage ledger that can be persisted by the artifact store."""

    def __init__(self) -> None:
        self._records: list[UsageEvent] = []

    @property
    def records(self) -> list[UsageEvent]:
        return list(self._records)

    def record_event(self, event: UsageEvent) -> None:
        self._records.append(event)

    def record_usage(
        self,
        context: RunContext,
        *,
        stage: str,
        kind: Literal["workflow", "model", "tool", "system"],
        usage: UsageRecord | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UsageEvent:
        usage = usage or UsageRecord()
        event = UsageEvent(
            experiment_id=context.experiment_id,
            run_id=context.run_id,
            system_id=context.system_id,
            corpus_id=context.corpus_id,
            item_id=context.item_id,
            repetition=context.repetition,
            stage=stage,
            kind=kind,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model_calls=usage.model_calls,
            tool_calls=usage.tool_calls,
            duration_ms=duration_ms if duration_ms is not None else usage.duration_ms,
            estimated_cost=usage.estimated_cost,
            metadata=metadata or {},
        )
        self.record_event(event)
        return event

    def summarize(self, run_id: str) -> UsageRecord:
        matching = [record for record in self._records if record.run_id == run_id]
        return summarize_usage(matching)


def summarize_usage(records: Iterable[UsageEvent]) -> UsageRecord:
    records = list(records)
    estimated_costs = [
        record.estimated_cost for record in records if record.estimated_cost is not None
    ]
    return UsageRecord(
        input_tokens=sum(record.input_tokens for record in records),
        output_tokens=sum(record.output_tokens for record in records),
        model_calls=sum(record.model_calls for record in records),
        tool_calls=sum(record.tool_calls for record in records),
        duration_ms=sum(record.duration_ms for record in records),
        estimated_cost=sum(estimated_costs) if estimated_costs else None,
    )
