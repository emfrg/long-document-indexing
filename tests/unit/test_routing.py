from __future__ import annotations

import json
from pathlib import Path

import pytest

from long_document_indexing.domain.maps import DocumentMap, MapEntry, SourceReference
from long_document_indexing.domain.runs import UsageRecord
from long_document_indexing.models.base import GenerationRequest, GenerationResponse
from long_document_indexing.models.structured_outputs import StructuredRoutingDecision
from long_document_indexing.prompts import PromptLoader
from long_document_indexing.retrieval.local_vector import LocalVectorBackend
from long_document_indexing.routing import route_documents_from_maps
from long_document_indexing.services import Services
from long_document_indexing.storage.artifacts import ArtifactStore
from long_document_indexing.telemetry.usage import UsageLedger
from long_document_indexing.workflows.execution import LocalWorkflowRunner


async def test_model_router_receives_structured_maps_and_returns_decision(tmp_path) -> None:
    generator = _RoutingClient(["doc_order", "doc_complaint"])
    progress: list[str] = []
    result = await route_documents_from_maps(
        query="What inquiry was skipped across the complaint and order?",
        document_maps=[_map("doc_complaint"), _map("doc_order")],
        services=_services(tmp_path, generator, progress=progress.append),
        top_k=2,
    )

    assert result.decision.selected_document_ids == ["doc_order", "doc_complaint"]
    assert result.decision.unresolved_information_needs == ["Confirm chronology"]
    assert result.usage.model_calls == 1
    assert generator.request is not None
    assert generator.request.response_model is StructuredRoutingDecision
    assert generator.request.metadata["document_maps"][0]["entries"][0]["id"] == "entry-1"
    assert "source_references" in generator.request.prompt
    assert progress == [
        "routing waiting for model response",
        "routing selected 2 document(s)",
    ]


async def test_model_router_rejects_unknown_document_id(tmp_path) -> None:
    generator = _RoutingClient(["invented_doc"])
    with pytest.raises(ValueError, match="unknown document"):
        await route_documents_from_maps(
            query="question",
            document_maps=[_map("doc_complaint")],
            services=_services(tmp_path, generator),
            top_k=1,
        )
    assert generator.calls == 3


async def test_model_router_recovers_from_unknown_document_id(tmp_path) -> None:
    generator = _RecoveringRoutingClient()
    progress: list[str] = []

    result = await route_documents_from_maps(
        query="question",
        document_maps=[_map("doc_complaint")],
        services=_services(tmp_path, generator, progress=progress.append),
        top_k=1,
    )

    assert result.decision.selected_document_ids == ["doc_complaint"]
    assert result.usage.model_calls == 2
    assert "exact identifiers" in generator.requests[1].prompt
    assert any("retrying with exact document identifiers (2/3)" in line for line in progress)


def test_structured_routing_content_does_not_include_domain_metadata() -> None:
    generated = StructuredRoutingDecision(
        selected_document_ids=["doc_complaint"],
        rationale="The complaint maps the relevant allegation.",
        unresolved_information_needs=[],
    )

    assert json.loads(generated.to_generation_content()) == {
        "selected_document_ids": ["doc_complaint"],
        "rationale": "The complaint maps the relevant allegation.",
        "unresolved_information_needs": [],
    }


class _RoutingClient:
    def __init__(self, selected_document_ids: list[str]) -> None:
        self.selected_document_ids = selected_document_ids
        self.request: GenerationRequest | None = None
        self.calls = 0

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        self.request = request
        return GenerationResponse(
            content=json.dumps(
                {
                    "selected_document_ids": self.selected_document_ids,
                    "rationale": "The mapped issues span the complaint and order.",
                    "unresolved_information_needs": ["Confirm chronology"],
                }
            ),
            usage=UsageRecord(input_tokens=20, output_tokens=8, model_calls=1),
        )


class _RecoveringRoutingClient:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        selected_document_ids = (
            ["invented_doc"] if len(self.requests) == 1 else ["doc_complaint"]
        )
        return GenerationResponse(
            content=json.dumps(
                {
                    "selected_document_ids": selected_document_ids,
                    "rationale": "The complaint is relevant.",
                    "unresolved_information_needs": [],
                }
            ),
            usage=UsageRecord(model_calls=1),
        )


def _map(document_id: str) -> DocumentMap:
    return DocumentMap(
        document_id=document_id,
        overview="Ability-to-pay procedure.",
        entries=[
            MapEntry(
                id="entry-1",
                kind="legal_issue",
                label="Ability-to-pay inquiry",
                summary="The document addresses an inquiry before imprisonment.",
                source_references=[
                    SourceReference(document_id=document_id, segment_ids=[f"{document_id}:seg-1"])
                ],
            )
        ],
        construction_method="test",
    )


def _services(tmp_path: Path, generator, *, progress=None) -> Services:
    store = ArtifactStore(tmp_path / "artifacts", "exp")
    return Services(
        artifact_store=store,
        retrieval_backend=LocalVectorBackend(store.path("indexes", "local_vector")),
        workflow_runner=LocalWorkflowRunner(),
        usage_ledger=UsageLedger(),
        prompt_loader=PromptLoader(Path("prompts")),
        generator_client=generator,
        progress=progress,
    )
