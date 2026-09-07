from __future__ import annotations

from long_document_indexing.config import SharedPipelineConfig
from long_document_indexing.domain.benchmark import BenchmarkItem
from long_document_indexing.domain.corpus import Corpus
from long_document_indexing.domain.maps import IndexArtifact
from long_document_indexing.domain.runs import RagRunRecord, RunContext, UsageRecord
from long_document_indexing.services import Services
from long_document_indexing.systems.base import RagSystem
from long_document_indexing.telemetry.tracing import stable_query_run_id
from long_document_indexing.workflows.execution import WorkflowExecutionError

WORKFLOW_NAME = "common_query"


async def run_query_workflow(
    *,
    system: RagSystem,
    item: BenchmarkItem,
    corpus: Corpus,
    index_artifact: IndexArtifact,
    services: Services,
    pipeline: SharedPipelineConfig,
    experiment_id: str,
    repetition: int,
) -> RagRunRecord:
    """Run the shared query workflow around a system-specific query executor."""

    context = RunContext(
        experiment_id=experiment_id,
        run_id=stable_query_run_id(experiment_id, system.id, item.id, repetition),
        system_id=system.id,
        corpus_id=corpus.id,
        item_id=item.id,
        repetition=repetition,
    )

    async def operation() -> RagRunRecord:
        record = await system.run_query(
            item,
            corpus,
            index_artifact,
            services,
            pipeline,
            experiment_id=experiment_id,
            repetition=repetition,
        )
        return validate_rag_run_record(
            record,
            context=context,
            system_id=system.id,
            corpus_id=corpus.id,
            item_id=item.id,
        )

    try:
        result = await services.workflow_runner.run(WORKFLOW_NAME, context, operation)
    except WorkflowExecutionError as exc:
        workflow_path = services.artifact_store.write_json(
            f"workflows/query/{system.id}/{item.id}/rep-{repetition}.json",
            exc.record,
        )
        services.usage_ledger.record_usage(
            context,
            stage=WORKFLOW_NAME,
            kind="workflow",
            duration_ms=exc.record.duration_ms,
            metadata={
                "workflow_artifact_path": str(workflow_path),
                "status": "failed",
            },
        )
        return RagRunRecord(
            run_id=context.run_id,
            experiment_id=context.experiment_id,
            system_id=system.id,
            corpus_id=corpus.id,
            item_id=item.id,
            repetition=repetition,
            selected_document_ids=[],
            retrieved_items=[],
            answer="",
            citations=[],
            usage=UsageRecord(),
            trace_id=exc.record.trace_id,
            workflow_artifact_path=str(workflow_path),
            status="failed",
            error=exc.record.error,
        )

    workflow_path = services.artifact_store.write_json(
        f"workflows/query/{system.id}/{item.id}/rep-{repetition}.json",
        result.record,
    )

    record = result.output.model_copy(
        update={
            "trace_id": result.record.trace_id,
            "workflow_artifact_path": str(workflow_path),
        }
    )
    services.usage_ledger.record_usage(
        context,
        stage="query_system",
        kind="system",
        usage=record.usage,
        metadata={"workflow_name": WORKFLOW_NAME},
    )
    services.usage_ledger.record_usage(
        context,
        stage=WORKFLOW_NAME,
        kind="workflow",
        duration_ms=result.record.duration_ms,
        metadata={"workflow_artifact_path": str(workflow_path)},
    )
    return record


def validate_rag_run_record(
    record: RagRunRecord,
    *,
    context: RunContext,
    system_id: str,
    corpus_id: str,
    item_id: str,
) -> RagRunRecord:
    if record.run_id != context.run_id:
        raise ValueError(f"run_id mismatch: expected {context.run_id!r}, got {record.run_id!r}")
    if record.experiment_id != context.experiment_id:
        raise ValueError(
            "experiment_id mismatch: "
            f"expected {context.experiment_id!r}, got {record.experiment_id!r}"
        )
    if record.system_id != system_id:
        raise ValueError(f"system_id mismatch: expected {system_id!r}, got {record.system_id!r}")
    if record.corpus_id != corpus_id:
        raise ValueError(f"corpus_id mismatch: expected {corpus_id!r}, got {record.corpus_id!r}")
    if record.item_id != item_id:
        raise ValueError(f"item_id mismatch: expected {item_id!r}, got {record.item_id!r}")
    if record.repetition != context.repetition:
        raise ValueError(
            f"repetition mismatch: expected {context.repetition}, got {record.repetition}"
        )
    return record
