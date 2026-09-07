from __future__ import annotations

import json
from pathlib import Path

import pytest

from long_document_indexing.answering import answer_from_retrieved_evidence
from long_document_indexing.domain.benchmark import BenchmarkItem, GroundTruth
from long_document_indexing.domain.runs import RetrievedItem, UsageRecord
from long_document_indexing.models.base import GenerationRequest, GenerationResponse
from long_document_indexing.models.fake import FakeTextGenerationClient
from long_document_indexing.models.structured_outputs import StructuredGeneratedAnswer
from long_document_indexing.prompts import PromptLoader
from long_document_indexing.retrieval.local_vector import LocalVectorBackend
from long_document_indexing.services import Services
from long_document_indexing.storage.artifacts import ArtifactStore
from long_document_indexing.telemetry.usage import UsageLedger
from long_document_indexing.workflows.execution import LocalWorkflowRunner


async def test_extractive_answer_uses_retrieved_items_without_model_call(tmp_path) -> None:
    services = _services(tmp_path, answering_mode="extractive")

    result = await answer_from_retrieved_evidence(
        item=_item(),
        retrieved_items=_retrieved_items(),
        services=services,
        extractive_prefix="Local answer",
    )

    assert result.answer.startswith("Local answer:")
    assert result.citations[0].document_id == "doc_alpha"
    assert result.citations[0].segment_id == "alpha_s1"
    assert result.usage.model_calls == 0


async def test_generated_answer_uses_structured_model_and_validates_citations(tmp_path) -> None:
    services = _services(
        tmp_path,
        answering_mode="generated",
        generator_client=FakeTextGenerationClient(),
    )

    result = await answer_from_retrieved_evidence(
        item=_item(),
        retrieved_items=_retrieved_items(),
        services=services,
        extractive_prefix="Local answer",
    )

    assert result.answer.startswith("Fake generated answer")
    assert [citation.segment_id for citation in result.citations] == [
        "alpha_s1",
        "alpha_s2",
    ]
    assert result.usage.model_calls == 1


async def test_generated_answer_rejects_citations_outside_retrieved_evidence(tmp_path) -> None:
    services = _services(
        tmp_path,
        answering_mode="generated",
        generator_client=_InvalidCitationClient(),
    )

    with pytest.raises(ValueError, match="unknown evidence"):
        await answer_from_retrieved_evidence(
            item=_item(),
            retrieved_items=_retrieved_items(),
            services=services,
            extractive_prefix="Local answer",
        )


class _InvalidCitationClient:
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        assert request.response_model is StructuredGeneratedAnswer
        return GenerationResponse(
            content=json.dumps(
                {
                    "status": "answered",
                    "answer": "Unsupported answer.",
                    "citations": [
                        {
                            "evidence_id": "evidence_999",
                            "document_id": "doc_alpha",
                            "segment_id": "alpha_s1",
                            "quote": "Unsupported quote.",
                        }
                    ],
                }
            ),
            usage=UsageRecord(model_calls=1),
        )


def _services(
    tmp_path: Path,
    *,
    answering_mode: str,
    generator_client=None,
) -> Services:
    store = ArtifactStore(tmp_path / "artifacts", "exp")
    return Services(
        artifact_store=store,
        retrieval_backend=LocalVectorBackend(store.path("indexes", "local_vector")),
        workflow_runner=LocalWorkflowRunner(),
        usage_ledger=UsageLedger(),
        prompt_loader=PromptLoader(Path("prompts")),
        answering_mode=answering_mode,
        generator_client=generator_client,
    )


def _item() -> BenchmarkItem:
    return BenchmarkItem(
        id="q_alpha",
        corpus_id="smoke-corpus",
        query="What did the board approve?",
        ground_truth=GroundTruth(),
    )


def _retrieved_items() -> list[RetrievedItem]:
    return [
        RetrievedItem(
            document_id="doc_alpha",
            segment_id="alpha_s1",
            text="The board approved the Alpha contract renewal.",
            rank=1,
            retrieval_stage="vector",
        ),
        RetrievedItem(
            document_id="doc_alpha",
            segment_id="alpha_s2",
            text="Finance estimated a twelve percent cost reduction.",
            rank=2,
            retrieval_stage="vector",
        ),
    ]
