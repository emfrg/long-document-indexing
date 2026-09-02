from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvidenceSpan(BaseModel):
    """Optional gold evidence pointer for datasets that support span-level labels."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    segment_id: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    quote: str | None = None


class GroundTruth(BaseModel):
    """Evaluator-only data that systems under test must not receive."""

    model_config = ConfigDict(extra="forbid")

    expected_answer: str | None = None
    relevant_document_ids: set[str] = Field(default_factory=set)
    relevant_segment_ids: set[str] = Field(default_factory=set)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    reference_summary: str | None = None


class BenchmarkItem(BaseModel):
    """A single benchmark query bound to one canonical corpus."""

    model_config = ConfigDict(extra="forbid")

    id: str
    corpus_id: str
    query: str
    ground_truth: GroundTruth
    tags: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "corpus_id", "query")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class DatasetCapabilities(BaseModel):
    """Declared annotation capabilities for metric compatibility checks."""

    model_config = ConfigDict(extra="forbid")

    has_expected_answers: bool = False
    has_relevant_documents: bool = False
    has_relevant_segments: bool = False
    has_evidence_spans: bool = False
    has_reference_summaries: bool = False


class QuestionSet(BaseModel):
    """Named collection of benchmark questions over one or more corpora."""

    model_config = ConfigDict(extra="forbid")

    id: str
    items: list[BenchmarkItem]
    capabilities: DatasetCapabilities = Field(default_factory=DatasetCapabilities)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_items(self) -> QuestionSet:
        if not self.items:
            raise ValueError("question set must contain at least one item")

        item_ids = [item.id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError(f"question set {self.id!r} contains duplicate item ids")

        return self
