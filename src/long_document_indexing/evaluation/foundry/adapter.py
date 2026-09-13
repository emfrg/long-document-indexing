from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from long_document_indexing.config import FoundryEvaluationConfig
from long_document_indexing.domain.benchmark import BenchmarkItem, EvidenceSpan, GroundTruth
from long_document_indexing.domain.runs import Citation, RagRunRecord, RetrievedItem
from long_document_indexing.storage.artifacts import ArtifactStore

FOUNDRY_EVALUATION_SCHEMA_VERSION = "foundry-evaluation-dataset/v2"


@dataclass(frozen=True)
class FoundryEvaluationExport:
    dataset_path: Path
    manifest_path: Path
    row_count: int
    dataset_sha256: str


class FoundryEvaluationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class FoundryEvaluationCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    segment_id: str | None = None
    quote: str | None = None


class FoundryRetrievedContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    segment_id: str | None = None
    rank: int
    text: str
    score: float | None = None
    retrieval_stage: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class FoundryRetrievedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    rank: int
    relevance_score: float = 0.0


class FoundryRetrievalGroundTruthDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    query_relevance_label: int = 4


class FoundryEvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    segment_id: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    quote: str | None = None


class FoundryExpectedBehavior(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_answer: str | None = None
    reference_summary: str | None = None
    relevant_document_ids: list[str] = Field(default_factory=list)
    relevant_segment_ids: list[str] = Field(default_factory=list)
    evidence: list[FoundryEvidenceSpan] = Field(default_factory=list)


class FoundryEvaluationRow(BaseModel):
    """One Foundry-ready single-turn evaluation row derived from a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    response: str
    context: str
    ground_truth: str
    messages: list[FoundryEvaluationMessage]
    retrieved_context: list[FoundryRetrievedContext]
    retrieved_documents: list[FoundryRetrievedDocument]
    retrieval_ground_truth: list[FoundryRetrievalGroundTruthDocument]
    citations: list[FoundryEvaluationCitation]
    expected_behavior: FoundryExpectedBehavior
    experiment_id: str
    run_id: str
    system_id: str
    corpus_id: str
    item_id: str
    repetition: int
    selected_document_ids: list[str]
    relevant_document_ids: list[str]
    relevant_segment_ids: list[str]
    tags: list[str]
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class FoundryEvaluationManifest(BaseModel):
    """Sidecar instructions for uploading or mapping the exported dataset in Foundry."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = FOUNDRY_EVALUATION_SCHEMA_VERSION
    experiment_id: str
    dataset_path: str
    dataset_sha256: str
    row_count: int
    systems: list[str]
    evaluation_level: Literal["turn"]
    evaluators: list[str]
    source_artifacts: list[str]
    item_schema: dict[str, Any]
    foundry_data_mapping: dict[str, str]
    evaluator_data_mappings: dict[str, dict[str, str]]
    azure_ai_evaluation_column_mapping: dict[str, str]
    notes: list[str]


def build_foundry_evaluation_rows(
    records: Iterable[RagRunRecord],
    items_by_id: Mapping[str, BenchmarkItem],
) -> list[FoundryEvaluationRow]:
    """Project normalized run records into Foundry JSONL dataset rows."""

    rows = []
    for record in sorted(records, key=_record_sort_key):
        item = items_by_id.get(record.item_id)
        if item is None:
            raise KeyError(f"run record references unknown benchmark item: {record.item_id!r}")
        rows.append(_build_row(record, item))
    return rows


def build_foundry_evaluation_manifest(
    *,
    experiment_id: str,
    dataset_path: Path,
    dataset_sha256: str,
    records: Iterable[RagRunRecord],
    rows: Iterable[FoundryEvaluationRow],
    config: FoundryEvaluationConfig,
) -> FoundryEvaluationManifest:
    row_list = list(rows)
    record_list = list(records)
    return FoundryEvaluationManifest(
        experiment_id=experiment_id,
        dataset_path=str(dataset_path),
        dataset_sha256=dataset_sha256,
        row_count=len(row_list),
        systems=sorted({row.system_id for row in row_list}),
        evaluation_level=config.evaluation_level,
        evaluators=config.evaluators,
        source_artifacts=sorted({f"runs/{record.system_id}.jsonl" for record in record_list}),
        item_schema=foundry_item_schema(),
        foundry_data_mapping=foundry_data_mapping(),
        evaluator_data_mappings=evaluator_data_mappings(config.evaluators),
        azure_ai_evaluation_column_mapping=azure_ai_evaluation_column_mapping(),
        notes=[
            "Rows contain precomputed benchmark responses; use dataset evaluation when "
            "you want Foundry to score these responses directly.",
            "Use model or agent targets only when you want Foundry to generate fresh "
            "responses from the query field.",
        ],
    )


def write_foundry_evaluation_export(
    *,
    store: ArtifactStore,
    records: Iterable[RagRunRecord],
    items_by_id: Mapping[str, BenchmarkItem],
    config: FoundryEvaluationConfig,
    experiment_id: str,
) -> FoundryEvaluationExport:
    record_list = list(records)
    rows = build_foundry_evaluation_rows(record_list, items_by_id)
    if not rows:
        raise ValueError("cannot export a Foundry evaluation dataset without run records")

    dataset_path = store.write_jsonl(config.dataset_path, rows)
    digest = dataset_sha256(dataset_path)
    manifest = build_foundry_evaluation_manifest(
        experiment_id=experiment_id,
        dataset_path=config.dataset_path,
        dataset_sha256=digest,
        records=record_list,
        rows=rows,
        config=config,
    )
    manifest_path = store.write_json(config.manifest_path, manifest)
    return FoundryEvaluationExport(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        row_count=len(rows),
        dataset_sha256=digest,
    )


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def foundry_data_mapping() -> dict[str, str]:
    return {
        "query": "{{item.query}}",
        "response": "{{item.response}}",
        "context": "{{item.context}}",
        "ground_truth": "{{item.ground_truth}}",
    }


def azure_ai_evaluation_column_mapping() -> dict[str, str]:
    return {
        "query": "${data.query}",
        "response": "${data.response}",
        "context": "${data.context}",
        "ground_truth": "${data.ground_truth}",
    }


def evaluator_data_mappings(evaluators: Iterable[str]) -> dict[str, dict[str, str]]:
    mapping_by_evaluator: dict[str, dict[str, str]] = {}
    for evaluator in evaluators:
        key = evaluator.lower()
        if key in {"groundedness", "groundedness_pro"}:
            mapping_by_evaluator[evaluator] = {
                "query": "{{item.query}}",
                "response": "{{item.response}}",
                "context": "{{item.context}}",
            }
        elif key == "relevance":
            mapping_by_evaluator[evaluator] = {
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            }
        elif key == "retrieval":
            mapping_by_evaluator[evaluator] = {
                "query": "{{item.query}}",
                "context": "{{item.context}}",
            }
        elif key == "response_completeness":
            mapping_by_evaluator[evaluator] = {
                "response": "{{item.response}}",
                "ground_truth": "{{item.ground_truth}}",
            }
        elif key == "document_retrieval":
            mapping_by_evaluator[evaluator] = {
                "retrieval_ground_truth": "{{item.retrieval_ground_truth}}",
                "retrieved_documents": "{{item.retrieved_documents}}",
            }
        else:
            mapping_by_evaluator[evaluator] = foundry_data_mapping()
    return mapping_by_evaluator


def foundry_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "query": {"type": "string"},
            "response": {"type": "string"},
            "context": {"type": "string"},
            "ground_truth": {"type": "string"},
            "messages": {"type": "array"},
            "retrieved_context": {"type": "array"},
            "retrieved_documents": {"type": "array"},
            "retrieval_ground_truth": {"type": "array"},
            "citations": {"type": "array"},
            "expected_behavior": {"type": "object"},
            "system_id": {"type": "string"},
            "corpus_id": {"type": "string"},
            "item_id": {"type": "string"},
            "run_id": {"type": "string"},
            "repetition": {"type": "integer"},
            "selected_document_ids": {"type": "array"},
            "relevant_document_ids": {"type": "array"},
            "relevant_segment_ids": {"type": "array"},
            "tags": {"type": "array"},
            "status": {"type": "string"},
            "metadata": {"type": "object"},
        },
        "required": ["id", "query", "response", "context", "ground_truth"],
    }


def _build_row(record: RagRunRecord, item: BenchmarkItem) -> FoundryEvaluationRow:
    truth = item.ground_truth
    relevant_document_ids = sorted(truth.relevant_document_ids)
    relevant_segment_ids = sorted(truth.relevant_segment_ids)

    return FoundryEvaluationRow(
        id=record.run_id,
        query=item.query,
        response=record.answer,
        context=_context_text(record.retrieved_items),
        ground_truth=_ground_truth_text(truth),
        messages=[
            FoundryEvaluationMessage(role="user", content=item.query),
            FoundryEvaluationMessage(role="assistant", content=record.answer),
        ],
        retrieved_context=[
            _retrieved_context(item) for item in sorted(record.retrieved_items, key=_rank_sort_key)
        ],
        retrieved_documents=_retrieved_documents(record),
        retrieval_ground_truth=[
            FoundryRetrievalGroundTruthDocument(document_id=document_id)
            for document_id in relevant_document_ids
        ],
        citations=[_citation(citation) for citation in record.citations],
        expected_behavior=FoundryExpectedBehavior(
            expected_answer=truth.expected_answer,
            reference_summary=truth.reference_summary,
            relevant_document_ids=relevant_document_ids,
            relevant_segment_ids=relevant_segment_ids,
            evidence=[_evidence_span(span) for span in truth.evidence],
        ),
        experiment_id=record.experiment_id,
        run_id=record.run_id,
        system_id=record.system_id,
        corpus_id=record.corpus_id,
        item_id=record.item_id,
        repetition=record.repetition,
        selected_document_ids=list(record.selected_document_ids),
        relevant_document_ids=relevant_document_ids,
        relevant_segment_ids=relevant_segment_ids,
        tags=sorted(item.tags),
        status=record.status,
        metadata={
            "error": record.error,
            "index_artifact_id": record.index_artifact_id,
            "index_artifact_signature": record.index_artifact_signature,
            "item_metadata": item.metadata,
            "query_policy_version": record.query_policy_version,
            "trace_id": record.trace_id,
            "usage": record.usage.model_dump(mode="json"),
            "workflow_artifact_path": record.workflow_artifact_path,
        },
    )


def _record_sort_key(record: RagRunRecord) -> tuple[str, str, int, str]:
    return record.system_id, record.item_id, record.repetition, record.run_id


def _rank_sort_key(item: RetrievedItem) -> tuple[int, str, str]:
    return item.rank, item.document_id, item.segment_id or ""


def _context_text(retrieved_items: list[RetrievedItem]) -> str:
    return "\n\n".join(_context_chunk(item) for item in sorted(retrieved_items, key=_rank_sort_key))


def _context_chunk(item: RetrievedItem) -> str:
    identifiers = f"rank={item.rank} document_id={item.document_id}"
    if item.segment_id is not None:
        identifiers = f"{identifiers} segment_id={item.segment_id}"
    return f"[{identifiers}]\n{item.text}"


def _ground_truth_text(truth: GroundTruth) -> str:
    return truth.expected_answer or truth.reference_summary or ""


def _retrieved_context(item: RetrievedItem) -> FoundryRetrievedContext:
    return FoundryRetrievedContext(
        document_id=item.document_id,
        segment_id=item.segment_id,
        rank=item.rank,
        text=item.text,
        score=item.score,
        retrieval_stage=item.retrieval_stage,
        metadata=item.metadata,
    )


def _retrieved_documents(record: RagRunRecord) -> list[FoundryRetrievedDocument]:
    selected_document_ids = record.selected_document_ids or _unique_document_ids(
        record.retrieved_items
    )
    scores_by_document = _first_score_by_document(record.retrieved_items)
    return [
        FoundryRetrievedDocument(
            document_id=document_id,
            rank=rank,
            relevance_score=scores_by_document.get(document_id, 0.0),
        )
        for rank, document_id in enumerate(selected_document_ids, start=1)
    ]


def _unique_document_ids(retrieved_items: list[RetrievedItem]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in sorted(retrieved_items, key=_rank_sort_key):
        if item.document_id in seen:
            continue
        ordered.append(item.document_id)
        seen.add(item.document_id)
    return ordered


def _first_score_by_document(retrieved_items: list[RetrievedItem]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in sorted(retrieved_items, key=_rank_sort_key):
        if item.score is None or item.document_id in scores:
            continue
        scores[item.document_id] = item.score
    return scores


def _citation(citation: Citation) -> FoundryEvaluationCitation:
    return FoundryEvaluationCitation(
        document_id=citation.document_id,
        segment_id=citation.segment_id,
        quote=citation.quote,
    )


def _evidence_span(span: EvidenceSpan) -> FoundryEvidenceSpan:
    return FoundryEvidenceSpan(
        document_id=span.document_id,
        segment_id=span.segment_id,
        start_char=span.start_char,
        end_char=span.end_char,
        quote=span.quote,
    )
