from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievedItem(BaseModel):
    """One retrieved source item in ranked order."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    segment_id: str | None = None
    text: str
    score: float | None = None
    rank: int
    retrieval_stage: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Answer citation normalized to corpus identifiers."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    segment_id: str | None = None
    quote: str | None = None


class UsageRecord(BaseModel):
    """Canonical usage/cost summary for one run."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    duration_ms: float = 0.0
    estimated_cost: float | None = None


class RunContext(BaseModel):
    """Stable identifiers attached to every workflow invocation."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    run_id: str
    system_id: str
    corpus_id: str
    item_id: str | None = None
    repetition: int = 0


class RagRunRecord(BaseModel):
    """Standardized system output consumed by all evaluators and reports."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    experiment_id: str
    system_id: str
    corpus_id: str
    item_id: str
    repetition: int = 0

    selected_document_ids: list[str]
    retrieved_items: list[RetrievedItem]
    answer: str
    citations: list[Citation]

    usage: UsageRecord = Field(default_factory=UsageRecord)
    index_artifact_id: str | None = None
    index_artifact_signature: str | None = None
    query_policy_version: str | None = None
    trace_id: str | None = None
    workflow_artifact_path: str | None = None
    status: Literal["succeeded", "failed", "skipped"]
    error: str | None = None


class MetricRecord(BaseModel):
    """One normalized metric value produced from a run record."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    run_id: str
    system_id: str
    corpus_id: str
    item_id: str
    repetition: int = 0
    level: str
    name: str
    value: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("level", "name")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value
