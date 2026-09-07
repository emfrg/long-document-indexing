from __future__ import annotations

import json
from typing import Any

from long_document_indexing.domain.maps import DocumentMap, MapEntry
from long_document_indexing.domain.runs import UsageRecord
from long_document_indexing.models.base import GenerationRequest, GenerationResponse
from long_document_indexing.text import approximate_token_count, summarize_text


class FakeTextGenerationClient:
    """Deterministic model replacement for local tests and smoke benchmarks."""

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        task = request.metadata.get("task")
        if task == "document_map":
            payload = _document_map_payload(request.metadata)
        elif task == "reduce_document_maps":
            payload = _reduce_document_maps_payload(request.metadata)
        elif task == "refine_document_map":
            payload = _refine_document_map_payload(request.metadata)
        elif task == "answer_query":
            payload = _answer_query_payload(request.metadata)
        else:
            payload = {"text": "Fake model response."}

        content = json.dumps(payload, sort_keys=True)
        return GenerationResponse(
            content=content,
            usage=UsageRecord(
                input_tokens=approximate_token_count(request.prompt),
                output_tokens=approximate_token_count(content),
                model_calls=1,
            ),
            metadata={"client": self.__class__.__name__, "task": task},
        )


def _document_map_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    document_id = str(metadata["document_id"])
    strategy = str(metadata["strategy"])
    segments = list(metadata.get("segments", []))
    text = " ".join(str(segment["text"]) for segment in segments)

    entries = [
        _entry_payload(
            document_id=document_id,
            segment_id=str(segment["id"]),
            order=int(segment["order"]),
            text=str(segment["text"]),
            prefix=strategy,
        )
        for segment in segments
    ]
    return DocumentMap(
        document_id=document_id,
        overview=summarize_text(text, max_chars=320),
        entries=[MapEntry.model_validate(entry) for entry in entries],
        facets={
            "segment_count": len(segments),
            "source_token_count": approximate_token_count(text),
        },
        construction_method=strategy,
    ).model_dump(mode="json")


def _reduce_document_maps_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    document_id = str(metadata["document_id"])
    strategy = str(metadata["strategy"])
    partial_maps = [DocumentMap.model_validate(item) for item in metadata.get("partial_maps", [])]
    entries = [
        entry
        for partial_map in partial_maps
        for root in partial_map.entries
        for entry in root.walk()
    ]
    overview = summarize_text(" ".join(partial_map.overview for partial_map in partial_maps))
    return DocumentMap(
        document_id=document_id,
        overview=overview,
        entries=entries,
        facets={"partial_map_count": len(partial_maps)},
        construction_method=strategy,
    ).model_dump(mode="json")


def _refine_document_map_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    existing_map = metadata.get("existing_map")
    next_payload = _document_map_payload(metadata)
    next_map = DocumentMap.model_validate(next_payload)
    if existing_map is None:
        return next_payload

    current_map = DocumentMap.model_validate(existing_map)
    known_ids = {entry.id for root in current_map.entries for entry in root.walk()}
    new_entries = [entry for entry in next_map.entries if entry.id not in known_ids]
    overview = summarize_text(f"{current_map.overview} {next_map.overview}", max_chars=320)
    return DocumentMap(
        document_id=current_map.document_id,
        overview=overview,
        entries=[*current_map.entries, *new_entries],
        facets={
            **current_map.facets,
            "refined_segment_count": len([*current_map.entries, *new_entries]),
        },
        construction_method=str(metadata["strategy"]),
    ).model_dump(mode="json")


def _answer_query_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    evidence = list(metadata.get("evidence", []))
    if not evidence:
        return {
            "status": "insufficient_evidence",
            "answer": "No retrieved evidence supports an answer.",
            "citations": [],
        }

    citations = [
        {
            "evidence_id": str(item["evidence_id"]),
            "document_id": str(item["document_id"]),
            "segment_id": str(item["segment_id"]),
            "quote": summarize_text(str(item["text"]), max_chars=180),
        }
        for item in evidence[:2]
    ]
    return {
        "status": "answered",
        "answer": "Fake generated answer from retrieved evidence: "
        + " ".join(citation["quote"] for citation in citations),
        "citations": citations,
    }


def _entry_payload(
    *,
    document_id: str,
    segment_id: str,
    order: int,
    text: str,
    prefix: str,
) -> dict[str, Any]:
    return {
        "id": f"{prefix}:{document_id}:entry_{order:04d}",
        "kind": "segment_summary",
        "label": _label(text, order),
        "summary": summarize_text(text),
        "source_references": [
            {
                "document_id": document_id,
                "segment_ids": [segment_id],
            }
        ],
        "children": [],
        "attributes": {"order": order},
    }


def _label(text: str, order: int) -> str:
    words = [word.strip(".,:;!?()[]{}") for word in text.split()]
    words = [word for word in words if word]
    if not words:
        return f"Segment {order}"
    return " ".join(words[:6])
