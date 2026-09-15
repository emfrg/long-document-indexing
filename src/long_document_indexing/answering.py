from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from long_document_indexing.domain.benchmark import BenchmarkItem
from long_document_indexing.domain.runs import Citation, RetrievedItem, UsageRecord
from long_document_indexing.models.base import GenerationRequest, TextGenerationClient
from long_document_indexing.models.structured_outputs import (
    StructuredAnswerCitation,
    StructuredGeneratedAnswer,
)
from long_document_indexing.progress import await_with_progress
from long_document_indexing.prompt_safety import (
    apply_prompt_safety_preamble,
    sanitize_for_model_prompt,
    sanitize_for_model_recovery_prompt,
)
from long_document_indexing.prompts import render_prompt
from long_document_indexing.services import Services

ANSWER_GENERATION_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: list[Citation]
    usage: UsageRecord = field(default_factory=UsageRecord)


async def answer_from_retrieved_evidence(
    *,
    item: BenchmarkItem,
    retrieved_items: list[RetrievedItem],
    services: Services,
    extractive_prefix: str,
) -> AnswerResult:
    if services.answering_mode == "extractive":
        return AnswerResult(
            answer=extractive_answer(retrieved_items, prefix=extractive_prefix),
            citations=citations_from_retrieved_items(retrieved_items),
        )
    if services.answering_mode == "generated":
        return await _generated_answer(
            item=item,
            retrieved_items=retrieved_items,
            services=services,
        )
    raise ValueError(f"unsupported answering mode: {services.answering_mode}")


def citations_from_retrieved_items(retrieved_items: list[RetrievedItem]) -> list[Citation]:
    return [
        Citation(
            document_id=item.document_id,
            segment_id=item.segment_id,
            quote=_quote(item.text),
        )
        for item in retrieved_items
    ]


def extractive_answer(retrieved_items: list[RetrievedItem], *, prefix: str) -> str:
    if not retrieved_items:
        return "No local evidence was retrieved."

    snippets = [
        f"[{item.document_id}/{item.segment_id}] {_quote(item.text, max_chars=180)}"
        for item in retrieved_items[:3]
    ]
    return f"{prefix}: " + " ".join(snippets)


def query_usage(
    answer_usage: UsageRecord,
    *,
    tool_calls: int,
    duration_ms: float,
) -> UsageRecord:
    return UsageRecord(
        input_tokens=answer_usage.input_tokens,
        output_tokens=answer_usage.output_tokens,
        model_calls=answer_usage.model_calls,
        tool_calls=tool_calls + answer_usage.tool_calls,
        duration_ms=duration_ms,
        estimated_cost=answer_usage.estimated_cost,
    )


async def _generated_answer(
    *,
    item: BenchmarkItem,
    retrieved_items: list[RetrievedItem],
    services: Services,
) -> AnswerResult:
    client = services.answer_client or services.generator_client
    if client is None:
        raise RuntimeError("generated answering requires a generator client")
    if not retrieved_items:
        return AnswerResult(answer="No local evidence was retrieved.", citations=[])

    evidence = _evidence_records(retrieved_items)
    prompt = apply_prompt_safety_preamble(
        render_prompt(
            services.prompt_loader.load("shared", "answer"),
            {
                "query": sanitize_for_model_prompt(item.query),
                "evidence": _evidence_for_prompt(evidence),
            },
        )
    )
    request = GenerationRequest(
        prompt=prompt,
        prompt_name="shared/answer",
        metadata={
            "task": "answer_query",
            "query": item.query,
            "evidence": evidence,
        },
        response_model=StructuredGeneratedAnswer,
    )
    generated, citations, usage = await _generate_answer_with_recovery(
        client=client,
        request=request,
        evidence=evidence,
        services=services,
    )
    return AnswerResult(
        answer=generated.answer,
        citations=citations,
        usage=usage,
    )


async def _generate_answer_with_recovery(
    *,
    client: TextGenerationClient,
    request: GenerationRequest,
    evidence: list[dict[str, Any]],
    services: Services,
) -> tuple[StructuredGeneratedAnswer, list[Citation], UsageRecord]:
    current_request = request
    usage_records: list[UsageRecord] = []
    for attempt in range(1, ANSWER_GENERATION_MAX_ATTEMPTS + 1):
        try:
            response = await await_with_progress(
                client.generate(current_request),
                emit=services.progress,
                message="answering waiting for model response",
            )
            usage_records.append(response.usage)
            generated = StructuredGeneratedAnswer.model_validate_json(response.content)
            if not generated.answer.strip():
                raise ValueError("generated answer is empty")
            citations = _validated_generated_citations(generated.citations, evidence)
        except Exception as exc:
            if attempt == ANSWER_GENERATION_MAX_ATTEMPTS or not _recoverable_answer_error(exc):
                raise
            services.emit_progress(
                "answering returned invalid or filtered structured output; "
                f"retrying with neutral legal abstraction ({attempt + 1}/"
                f"{ANSWER_GENERATION_MAX_ATTEMPTS})"
            )
            current_request = request.model_copy(
                update={"prompt": _answer_recovery_prompt(request.prompt)}
            )
            continue
        return generated, citations, _combine_usage_records(usage_records)

    raise AssertionError("answer generation recovery loop terminated unexpectedly")


def _answer_recovery_prompt(prompt: str) -> str:
    return (
        f"{sanitize_for_model_recovery_prompt(prompt)}\n\n"
        "Validation correction: return one complete structured answer. Use only the exact "
        "evidence_id values supplied in the prompt, keep the answer under 250 words, and keep "
        "each citation quote under 40 words. If the evidence cannot support an answer, return "
        "the structured insufficient_evidence result."
    )


def _recoverable_answer_error(exc: Exception) -> bool:
    if isinstance(exc, ValidationError):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "cannot assist",
            "can't assist",
            "content_filter",
            "generated answer is empty",
            "invalid json",
            "parsed structured output",
            "response ended with status incomplete",
            "unable to assist",
            "unknown evidence",
        )
    )


def _combine_usage_records(records: list[UsageRecord]) -> UsageRecord:
    estimated_costs = [
        record.estimated_cost for record in records if record.estimated_cost is not None
    ]
    return UsageRecord(
        input_tokens=sum(record.input_tokens for record in records),
        output_tokens=sum(record.output_tokens for record in records),
        model_calls=sum(record.model_calls for record in records),
        tool_calls=sum(record.tool_calls for record in records),
        duration_ms=sum(record.duration_ms for record in records),
        estimated_cost=sum(estimated_costs) if estimated_costs else None,
    )


def _validated_generated_citations(
    citations: list[StructuredAnswerCitation],
    evidence: list[dict[str, Any]],
) -> list[Citation]:
    evidence_by_id = {str(item["evidence_id"]): item for item in evidence}
    validated: list[Citation] = []

    for citation in citations:
        source = evidence_by_id.get(citation.evidence_id)
        if source is None:
            raise ValueError(f"generated answer cited unknown evidence: {citation.evidence_id}")
        validated.append(
            Citation(
                document_id=str(source["document_id"]),
                segment_id=str(source["segment_id"]),
                quote=citation.quote,
            )
        )

    return validated


def _evidence_records(retrieved_items: list[RetrievedItem]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": f"evidence_{index + 1}",
            "document_id": item.document_id,
            "segment_id": item.segment_id or "",
            "rank": item.rank,
            "text": item.text,
        }
        for index, item in enumerate(retrieved_items)
    ]


def _evidence_for_prompt(evidence: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        "\n".join(
            [
                f"[{item['evidence_id']}]",
                f"document_id: {item['document_id']}",
                f"segment_id: {item['segment_id']}",
                f"rank: {item['rank']}",
                f"text: {sanitize_for_model_prompt(str(item['text']))}",
            ]
        )
        for item in evidence
    )


def _quote(text: str, max_chars: int = 240) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 3]}..."
