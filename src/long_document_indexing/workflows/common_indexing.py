from __future__ import annotations

from long_document_indexing.config import SharedPipelineConfig
from long_document_indexing.domain.corpus import Corpus
from long_document_indexing.domain.maps import IndexArtifact
from long_document_indexing.domain.runs import RunContext, UsageRecord
from long_document_indexing.services import Services
from long_document_indexing.systems.base import RagSystem
from long_document_indexing.telemetry.tracing import stable_index_run_id
from long_document_indexing.workflows.execution import WorkflowExecutionError

WORKFLOW_NAME = "common_indexing"


async def run_indexing_workflow(
    *,
    system: RagSystem,
    corpus: Corpus,
    services: Services,
    pipeline: SharedPipelineConfig,
    experiment_id: str,
) -> IndexArtifact:
    """Run the shared indexing workflow around a system-specific indexer."""

    context = RunContext(
        experiment_id=experiment_id,
        run_id=stable_index_run_id(experiment_id, system.id, corpus.id),
        system_id=system.id,
        corpus_id=corpus.id,
    )

    async def operation() -> IndexArtifact:
        artifact = await system.build_index(corpus, services, pipeline)
        return validate_index_artifact(artifact, system_id=system.id, corpus_id=corpus.id)

    try:
        result = await services.workflow_runner.run(WORKFLOW_NAME, context, operation)
    except WorkflowExecutionError as exc:
        workflow_path = services.artifact_store.write_json(
            f"workflows/indexing/{system.id}/{corpus.id}.json",
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
        raise

    workflow_path = services.artifact_store.write_json(
        f"workflows/indexing/{system.id}/{corpus.id}.json",
        result.record,
    )
    usage_payload = result.output.build_metadata.get("usage")
    if isinstance(usage_payload, dict):
        services.usage_ledger.record_usage(
            context,
            stage="index_system",
            kind="system",
            usage=UsageRecord.model_validate(usage_payload),
            metadata={"workflow_name": WORKFLOW_NAME},
        )
    services.usage_ledger.record_usage(
        context,
        stage=WORKFLOW_NAME,
        kind="workflow",
        duration_ms=result.record.duration_ms,
        metadata={"workflow_artifact_path": str(workflow_path)},
    )

    artifact = result.output
    return artifact.model_copy(
        update={
            "build_metadata": {
                **artifact.build_metadata,
                "workflow_trace_id": result.record.trace_id,
                "workflow_duration_ms": result.record.duration_ms,
                "workflow_artifact_path": str(workflow_path),
            }
        }
    )


def validate_index_artifact(
    artifact: IndexArtifact,
    *,
    system_id: str,
    corpus_id: str,
) -> IndexArtifact:
    if artifact.system_id != system_id:
        raise ValueError(
            f"index artifact system_id mismatch: expected {system_id!r}, got {artifact.system_id!r}"
        )
    if artifact.corpus_id != corpus_id:
        raise ValueError(
            f"index artifact corpus_id mismatch: expected {corpus_id!r}, got {artifact.corpus_id!r}"
        )
    if not artifact.id.strip():
        raise ValueError("index artifact id must not be blank")
    if not artifact.artifact_path.strip():
        raise ValueError("index artifact artifact_path must not be blank")
    return artifact
