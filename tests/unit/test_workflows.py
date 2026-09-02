from __future__ import annotations

from pathlib import Path

import pytest

from long_document_indexing.config import SharedPipelineConfig
from long_document_indexing.domain.benchmark import BenchmarkItem, GroundTruth
from long_document_indexing.domain.corpus import Corpus, Document, Segment
from long_document_indexing.domain.maps import IndexArtifact
from long_document_indexing.prompts import PromptLoader
from long_document_indexing.retrieval.local_vector import LocalVectorBackend
from long_document_indexing.services import Services
from long_document_indexing.storage.artifacts import ArtifactStore
from long_document_indexing.systems.flat_vector import FlatVectorSystem
from long_document_indexing.telemetry.usage import UsageLedger
from long_document_indexing.workflows.common_indexing import (
    run_indexing_workflow,
    validate_index_artifact,
)
from long_document_indexing.workflows.common_query import run_query_workflow
from long_document_indexing.workflows.execution import LocalWorkflowRunner


async def test_common_workflows_persist_records_and_usage(tmp_path) -> None:
    corpus = _corpus()
    item = BenchmarkItem(
        id="q1",
        corpus_id=corpus.id,
        query="alpha board approval",
        ground_truth=GroundTruth(
            relevant_document_ids={"doc_alpha"},
            relevant_segment_ids={"alpha_s1"},
        ),
    )
    services = _services(tmp_path)
    system = FlatVectorSystem()
    pipeline = SharedPipelineConfig(selected_documents=1, retrieved_segments=2)

    artifact = await run_indexing_workflow(
        system=system,
        corpus=corpus,
        services=services,
        pipeline=pipeline,
        experiment_id="exp",
    )
    record = await run_query_workflow(
        system=system,
        item=item,
        corpus=corpus,
        index_artifact=artifact,
        services=services,
        pipeline=pipeline,
        experiment_id="exp",
        repetition=0,
    )

    assert artifact.build_metadata["workflow_trace_id"].startswith("trace-")
    assert record.trace_id is not None
    assert record.trace_id.startswith("trace-")
    assert record.workflow_artifact_path is not None
    assert Path(record.workflow_artifact_path).exists()
    assert {event.stage for event in services.usage_ledger.records} == {
        "common_indexing",
        "query_system",
        "common_query",
    }


def test_validate_index_artifact_rejects_mismatched_system() -> None:
    artifact = IndexArtifact(
        id="idx",
        system_id="wrong",
        corpus_id="corpus",
        artifact_path="artifact",
    )

    with pytest.raises(ValueError, match="system_id mismatch"):
        validate_index_artifact(artifact, system_id="flat_vector", corpus_id="corpus")


def _services(tmp_path) -> Services:
    store = ArtifactStore(tmp_path / "artifacts", "exp")
    prompt_root = tmp_path / "prompts"
    prompt_root.mkdir()
    return Services(
        artifact_store=store,
        retrieval_backend=LocalVectorBackend(store.path("indexes", "local_vector")),
        workflow_runner=LocalWorkflowRunner(),
        usage_ledger=UsageLedger(),
        prompt_loader=PromptLoader(prompt_root),
    )


def _corpus() -> Corpus:
    return Corpus(
        id="corpus",
        documents=[
            Document(
                id="doc_alpha",
                corpus_id="corpus",
                segments=[
                    Segment(
                        id="alpha_s1",
                        document_id="doc_alpha",
                        order=1,
                        text="Alpha renewal approval from the board.",
                    )
                ],
            )
        ],
    )
