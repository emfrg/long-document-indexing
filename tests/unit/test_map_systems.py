from __future__ import annotations

from pathlib import Path

import pytest

from long_document_indexing.config import SharedPipelineConfig
from long_document_indexing.domain.benchmark import BenchmarkItem, GroundTruth
from long_document_indexing.domain.corpus import Corpus, Document, Segment
from long_document_indexing.domain.maps import DocumentMap, MapEntry, SourceReference
from long_document_indexing.domain.runs import UsageRecord
from long_document_indexing.evaluation.local.maps import evaluate_index_artifact
from long_document_indexing.models.base import GenerationRequest, GenerationResponse
from long_document_indexing.models.fake import FakeTextGenerationClient
from long_document_indexing.prompts import PromptLoader
from long_document_indexing.retrieval.local_vector import LocalVectorBackend
from long_document_indexing.services import Services
from long_document_indexing.storage.artifacts import ArtifactStore
from long_document_indexing.storage.maps import read_document_map
from long_document_indexing.systems.agentic_map import AgenticMapSystem
from long_document_indexing.systems.hierarchical_map import HierarchicalMapSystem
from long_document_indexing.systems.map_base import DocumentMapSystemBase, MapBuildResult
from long_document_indexing.systems.map_reduce import MapReduceSystem
from long_document_indexing.systems.outline_then_fill import OutlineThenFillSystem
from long_document_indexing.systems.refine import RefineSystem
from long_document_indexing.systems.stuffing import StuffingSystem
from long_document_indexing.telemetry.usage import UsageLedger
from long_document_indexing.workflows.common_indexing import run_indexing_workflow
from long_document_indexing.workflows.common_query import run_query_workflow
from long_document_indexing.workflows.execution import LocalWorkflowRunner, WorkflowExecutionError


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

    for system in [
        StuffingSystem(),
        MapReduceSystem(),
        RefineSystem(),
        HierarchicalMapSystem(branching_factor=2),
        OutlineThenFillSystem(),
        AgenticMapSystem(),
    ]:
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


async def test_advanced_map_systems_record_strategy_metadata(tmp_path) -> None:
    corpus = _corpus()
    services = _services(tmp_path)
    pipeline = SharedPipelineConfig(selected_documents=1, retrieved_segments=2)

    systems = [
        HierarchicalMapSystem(branching_factor=2),
        OutlineThenFillSystem(),
        AgenticMapSystem(),
    ]

    for system in systems:
        artifact = await run_indexing_workflow(
            system=system,
            corpus=corpus,
            services=services,
            pipeline=pipeline,
            experiment_id="exp",
        )

        assert len(artifact.document_map_ids) == 2
        assert artifact.build_metadata["intermediate_map_paths"]
        assert artifact.build_metadata["strategy_metadata"]["documents"]


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


async def test_mapped_query_can_generate_model_backed_answer(tmp_path) -> None:
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
    services = _services(tmp_path, answering_mode="generated")
    system = StuffingSystem()

    artifact = await run_indexing_workflow(
        system=system,
        corpus=corpus,
        services=services,
        pipeline=SharedPipelineConfig(selected_documents=1, retrieved_segments=2),
        experiment_id="exp",
    )
    record = await run_query_workflow(
        system=system,
        item=item,
        corpus=corpus,
        index_artifact=artifact,
        services=services,
        pipeline=SharedPipelineConfig(selected_documents=1, retrieved_segments=2),
        experiment_id="exp",
        repetition=0,
    )

    assert record.answer.startswith("Fake generated answer")
    assert record.usage.model_calls == 1
    assert {citation.segment_id for citation in record.citations} <= {
        retrieved.segment_id for retrieved in record.retrieved_items
    }


async def test_map_build_normalizes_invalid_source_references_before_persistence(
    tmp_path,
) -> None:
    corpus = _corpus()
    services = _services(tmp_path)
    system = _BadReferenceSystem()

    artifact = await run_indexing_workflow(
        system=system,
        corpus=corpus,
        services=services,
        pipeline=SharedPipelineConfig(selected_documents=1, retrieved_segments=2),
        experiment_id="exp",
    )

    map_path = next(iter(artifact.build_metadata["document_map_paths"].values()))
    document_map = read_document_map(map_path)
    references = document_map.entries[0].source_references
    metrics = evaluate_index_artifact(
        artifact,
        corpus,
        services.artifact_store,
        ["map_source_reference_validity"],
        experiment_id="exp",
    )

    assert references == [SourceReference(document_id="doc_alpha", segment_ids=["alpha_s1"])]
    assert document_map.entries[0].children[0].source_references == [
        SourceReference(document_id="doc_alpha", segment_ids=["alpha_s2"])
    ]
    assert document_map.facets["source_reference_normalization"] == {
        "map_document_id_rewrites": 0,
        "reference_document_id_rewrites": 1,
        "references_dropped": 2,
        "invalid_segment_ids_dropped": 2,
        "duplicate_segment_ids_dropped": 1,
        "entries_without_source_references": 0,
        "wrong_reference_document_ids": ["doc_wrong"],
        "invalid_segment_ids": ["missing_segment", "beta_s1"],
    }
    assert artifact.build_metadata["source_reference_normalization"]["repaired_map_count"] == 1
    assert metrics[0].value == 1.0


async def test_map_reduce_reuses_partial_checkpoints_after_failed_indexing(tmp_path) -> None:
    corpus = _single_document_corpus()
    first_client = _FlakyMapReduceClient(fail_on_document_map_call=2)
    first_services = _services(tmp_path, generator_client=first_client)

    with pytest.raises(WorkflowExecutionError, match="planned map failure"):
        await run_indexing_workflow(
            system=MapReduceSystem(),
            corpus=corpus,
            services=first_services,
            pipeline=SharedPipelineConfig(selected_documents=1, retrieved_segments=2),
            experiment_id="exp",
        )

    checkpoint_dir = tmp_path / "artifacts" / "exp" / "indexes" / "map_reduce" / "checkpoints"
    assert len(list(checkpoint_dir.glob("*/completed/*.json"))) == 1
    assert len(list(checkpoint_dir.glob("*/failures/*.json"))) == 1
    assert first_client.document_map_segment_ids == ["alpha_s1", "alpha_s2"]

    progress_messages: list[str] = []
    second_client = _FlakyMapReduceClient()
    second_services = _services(
        tmp_path,
        generator_client=second_client,
        resume_checkpoints=True,
        progress=progress_messages.append,
    )

    artifact = await run_indexing_workflow(
        system=MapReduceSystem(),
        corpus=corpus,
        services=second_services,
        pipeline=SharedPipelineConfig(selected_documents=1, retrieved_segments=2),
        experiment_id="exp",
    )

    assert second_client.document_map_segment_ids == ["alpha_s2"]
    assert second_client.reduce_calls == 1
    assert artifact.build_metadata["usage"]["model_calls"] == 3
    assert artifact.build_metadata["strategy_metadata"]["checkpointing"] == {
        "partial_reused": 1,
        "partial_written": 1,
        "reduction_reused": 0,
        "reduction_written": 1,
    }
    assert any("reused checkpoint" in message for message in progress_messages)


def _services(
    tmp_path,
    *,
    answering_mode: str = "extractive",
    generator_client=None,
    resume_checkpoints: bool = False,
    progress=None,
) -> Services:
    store = ArtifactStore(tmp_path / "artifacts", "exp")
    return Services(
        artifact_store=store,
        retrieval_backend=LocalVectorBackend(store.path("indexes", "local_vector")),
        workflow_runner=LocalWorkflowRunner(),
        usage_ledger=UsageLedger(),
        prompt_loader=PromptLoader(Path("prompts")),
        answering_mode=answering_mode,
        generator_client=generator_client or FakeTextGenerationClient(),
        resume_checkpoints=resume_checkpoints,
        progress=progress,
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


def _single_document_corpus() -> Corpus:
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
        ],
    )


class _FlakyMapReduceClient:
    def __init__(self, fail_on_document_map_call: int | None = None) -> None:
        self.fail_on_document_map_call = fail_on_document_map_call
        self.document_map_segment_ids: list[str] = []
        self.reduce_calls = 0
        self._fake = FakeTextGenerationClient()

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        task = request.metadata.get("task")
        if task == "document_map":
            segment_id = str(request.metadata["segments"][0]["id"])
            self.document_map_segment_ids.append(segment_id)
            if len(self.document_map_segment_ids) == self.fail_on_document_map_call:
                raise RuntimeError("planned map failure")
        elif task == "reduce_document_maps":
            self.reduce_calls += 1
        return await self._fake.generate(request)


class _BadReferenceSystem(DocumentMapSystemBase):
    id = "bad_reference"
    construction_method = "bad_reference"

    async def build_document_maps(
        self,
        corpus: Corpus,
        services: Services,
        pipeline: SharedPipelineConfig,
    ) -> MapBuildResult:
        del services, pipeline
        return MapBuildResult(
            document_maps=[
                DocumentMap(
                    document_id="doc_alpha",
                    overview="Bad reference map.",
                    entries=[
                        MapEntry(
                            id="entry_1",
                            kind="section",
                            label="Alpha",
                            summary="Alpha summary.",
                            source_references=[
                                SourceReference(
                                    document_id="doc_wrong",
                                    segment_ids=[
                                        "alpha_s1",
                                        "missing_segment",
                                        "alpha_s1",
                                    ],
                                ),
                                SourceReference(
                                    document_id="doc_alpha",
                                    segment_ids=["beta_s1"],
                                ),
                                SourceReference(
                                    document_id="doc_alpha",
                                    segment_ids=[],
                                ),
                            ],
                            children=[
                                MapEntry(
                                    id="entry_2",
                                    kind="section",
                                    label="Alpha child",
                                    summary="Alpha child summary.",
                                    source_references=[
                                        SourceReference(
                                            document_id="doc_alpha",
                                            segment_ids=["alpha_s2"],
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                    construction_method=self.construction_method,
                )
            ],
            usage=UsageRecord(),
            statuses={"doc_alpha": "indexed"},
        )
