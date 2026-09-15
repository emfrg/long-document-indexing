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
    progress: list[str] = []
    services = _services(
        tmp_path,
        answering_mode="generated",
        generator_client=FakeTextGenerationClient(),
        progress=progress.append,
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
    assert progress == ["answering waiting for model response"]


async def test_generated_answer_rejects_citations_outside_retrieved_evidence(tmp_path) -> None:
    client = _InvalidCitationClient()
    services = _services(
        tmp_path,
        answering_mode="generated",
        generator_client=client,
    )

    with pytest.raises(ValueError, match="unknown evidence"):
        await answer_from_retrieved_evidence(
            item=_item(),
            retrieved_items=_retrieved_items(),
            services=services,
            extractive_prefix="Local answer",
        )
    assert client.calls == 3


async def test_generated_answer_recovers_from_sensitive_content_refusal(tmp_path) -> None:
    progress: list[str] = []
    client = _RefusalThenValidClient()
    services = _services(
        tmp_path,
        answering_mode="generated",
        generator_client=client,
        progress=progress.append,
    )

    result = await answer_from_retrieved_evidence(
        item=_item(),
        retrieved_items=_retrieved_items(),
        services=services,
        extractive_prefix="Local answer",
    )

    assert result.answer.startswith("Fake generated answer")
    assert result.usage.model_calls == 2
    assert client.prompts[1].startswith("Recovery instruction:")
    assert any("retrying with neutral legal abstraction (2/3)" in line for line in progress)


async def test_generated_answer_recovers_from_truncated_json(tmp_path) -> None:
    client = _RefusalThenValidClient(first_content='{"status":"answered","answer":"cut off')
    services = _services(
        tmp_path,
        answering_mode="generated",
        generator_client=client,
    )

    result = await answer_from_retrieved_evidence(
        item=_item(),
        retrieved_items=_retrieved_items(),
        services=services,
        extractive_prefix="Local answer",
    )

    assert result.answer.startswith("Fake generated answer")
    assert result.usage.model_calls == 2


async def test_generated_answer_normalizes_citation_ids_from_evidence_id(tmp_path) -> None:
    services = _services(
        tmp_path,
        answering_mode="generated",
        generator_client=_MismatchedCitationClient(),
    )

    result = await answer_from_retrieved_evidence(
        item=_item(),
        retrieved_items=_retrieved_items(),
        services=services,
        extractive_prefix="Local answer",
    )

    assert result.citations[0].document_id == "doc_alpha"
    assert result.citations[0].segment_id == "alpha_s2"
    assert result.citations[0].quote == "Supported quote."


class _InvalidCitationClient:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
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


class _RefusalThenValidClient:
    def __init__(
        self,
        first_content: str = "I'm sorry, but I cannot assist with that request.",
    ) -> None:
        self.prompts: list[str] = []
        self.first_content = first_content
        self._fake = FakeTextGenerationClient()

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.prompts.append(request.prompt)
        if len(self.prompts) == 1:
            return GenerationResponse(
                content=self.first_content,
                usage=UsageRecord(model_calls=1),
            )
        return await self._fake.generate(request)


class _MismatchedCitationClient:
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        assert request.response_model is StructuredGeneratedAnswer
        return GenerationResponse(
            content=json.dumps(
                {
                    "status": "answered",
                    "answer": "Supported answer.",
                    "citations": [
                        {
                            "evidence_id": "evidence_2",
                            "document_id": "wrong_doc",
                            "segment_id": "wrong_segment",
                            "quote": "Supported quote.",
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
        generator_client=generator_client,
        progress=progress,
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
