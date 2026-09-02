from __future__ import annotations

from pathlib import Path

from long_document_indexing.config import SharedPipelineConfig
from long_document_indexing.domain.benchmark import BenchmarkItem, GroundTruth
from long_document_indexing.domain.corpus import Corpus, Document, Segment
from long_document_indexing.models.fake import FakeTextGenerationClient
from long_document_indexing.prompts import PromptLoader
from long_document_indexing.retrieval.local_vector import LocalVectorBackend
from long_document_indexing.services import Services
from long_document_indexing.storage.artifacts import ArtifactStore
from long_document_indexing.systems.map_reduce import MapReduceSystem
from long_document_indexing.systems.refine import RefineSystem
from long_document_indexing.systems.stuffing import StuffingSystem
from long_document_indexing.telemetry.usage import UsageLedger
from long_document_indexing.workflows.common_indexing import run_indexing_workflow
from long_document_indexing.workflows.common_query import run_query_workflow
from long_document_indexing.workflows.execution import LocalWorkflowRunner


async def test_map_systems_build_maps_and_answer_queries(tmp_path) -> None:
    corpus = _corpus()
    item = BenchmarkItem(
        id="q_alpha",
        corpus_id=corpus.id,
        query="alpha board approval",
        ground_truth=GroundTruth(
            relevant_document_ids={"doc_alpha"},
            relevant_segment_ids={"alpha_s1"},
        ),
    )
    services = _services(tmp_path)
    pipeline = SharedPipelineConfig(selected_documents=1, retrieved_segments=2)

    for system in [StuffingSystem(), MapReduceSystem(), RefineSystem()]:
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

        assert len(artifact.document_map_ids) == 2
        assert record.selected_document_ids == ["doc_alpha"]
        assert record.retrieved_items[0].segment_id == "alpha_s1"


async def test_stuffing_marks_context_overflow(tmp_path) -> None:
    corpus = _corpus()
    services = _services(tmp_path)
    system = StuffingSystem()

    artifact = await run_indexing_workflow(
        system=system,
        corpus=corpus,
        services=services,
        pipeline=SharedPipelineConfig(segment_tokens=2),
        experiment_id="exp",
    )

    assert artifact.document_map_ids == []
    assert set(artifact.build_metadata["document_statuses"].values()) == {"context_overflow"}


def _services(tmp_path) -> Services:
    store = ArtifactStore(tmp_path / "artifacts", "exp")
    return Services(
        artifact_store=store,
        retrieval_backend=LocalVectorBackend(store.path("indexes", "local_vector")),
        workflow_runner=LocalWorkflowRunner(),
        usage_ledger=UsageLedger(),
        prompt_loader=PromptLoader(Path("prompts")),
        generator_client=FakeTextGenerationClient(),
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
                    ),
                    Segment(
                        id="alpha_s2",
                        document_id="doc_alpha",
                        order=2,
                        text="Finance expected support cost reductions.",
                    ),
                ],
            ),
            Document(
                id="doc_beta",
                corpus_id="corpus",
                segments=[
                    Segment(
                        id="beta_s1",
                        document_id="doc_beta",
                        order=1,
                        text="Beta storage incident timeline.",
                    )
                ],
            ),
        ],
    )
