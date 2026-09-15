from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from long_document_indexing.domain.maps import DocumentMap
from long_document_indexing.domain.runs import RoutingDecision, UsageRecord
from long_document_indexing.models.base import GenerationRequest, TextGenerationClient
from long_document_indexing.models.structured_outputs import StructuredRoutingDecision
from long_document_indexing.progress import await_with_progress
from long_document_indexing.prompt_safety import (
    apply_prompt_safety_preamble,
    sanitize_for_model_prompt,
    sanitize_for_model_recovery_prompt,
    sanitize_prompt_payload,
)
from long_document_indexing.prompts import render_prompt
from long_document_indexing.services import Services

ROUTING_POLICY_VERSION = "model-map-router/v1"
ROUTING_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class RoutingResult:
    decision: RoutingDecision
    usage: UsageRecord


async def route_documents_from_maps(
    *,
    query: str,
    document_maps: list[DocumentMap],
    services: Services,
    top_k: int,
) -> RoutingResult:
    client = services.router_client or services.generator_client
    if client is None:
        raise RuntimeError("document-map routing requires a generator client")
    if top_k < 1:
        return RoutingResult(
            decision=RoutingDecision(
                selected_document_ids=[],
                rationale="The configured document limit is zero.",
                unresolved_information_needs=[query],
            ),
            usage=UsageRecord(),
        )
    if not document_maps:
        raise ValueError("document-map routing requires at least one map")

    maps_payload = [document_map.model_dump(mode="json") for document_map in document_maps]
    prompt = apply_prompt_safety_preamble(
        render_prompt(
            services.prompt_loader.load("shared", "route"),
            {
                "query": sanitize_for_model_prompt(query),
                "max_documents": top_k,
                "document_maps": json.dumps(
                    sanitize_prompt_payload(maps_payload),
                    indent=2,
                    sort_keys=True,
                ),
            },
        )
    )
    request = GenerationRequest(
        prompt=prompt,
        prompt_name="shared/route",
        metadata={
            "task": "route_documents",
            "query": query,
            "max_documents": top_k,
            "document_maps": maps_payload,
            "routing_policy": ROUTING_POLICY_VERSION,
        },
        response_model=StructuredRoutingDecision,
    )
    known_document_ids = {document_map.document_id for document_map in document_maps}
    generated, selected_document_ids, response_metadata, usage = (
        await _generate_routing_decision_with_recovery(
            client=client,
            request=request,
            known_document_ids=known_document_ids,
            top_k=top_k,
            services=services,
        )
    )

    services.emit_progress(f"routing selected {len(selected_document_ids)} document(s)")

    return RoutingResult(
        decision=RoutingDecision(
            selected_document_ids=selected_document_ids,
            rationale=generated.rationale.strip(),
            unresolved_information_needs=[
                need.strip()
                for need in generated.unresolved_information_needs
                if need.strip()
            ],
            model_id=_response_model_id(response_metadata),
        ),
        usage=usage,
    )


async def _generate_routing_decision_with_recovery(
    *,
    client: TextGenerationClient,
    request: GenerationRequest,
    known_document_ids: set[str],
    top_k: int,
    services: Services,
) -> tuple[StructuredRoutingDecision, list[str], dict, UsageRecord]:
    current_request = request
    usage_records: list[UsageRecord] = []
    for attempt in range(1, ROUTING_MAX_ATTEMPTS + 1):
        try:
            response = await await_with_progress(
                client.generate(current_request),
                emit=services.progress,
                message="routing waiting for model response",
            )
            usage_records.append(response.usage)
            generated = StructuredRoutingDecision.model_validate_json(response.content)
            selected_document_ids = _validated_document_ids(
                generated.selected_document_ids,
                known_document_ids=known_document_ids,
                top_k=top_k,
            )
            if not selected_document_ids:
                raise ValueError("map router did not select any known document identifiers")
        except Exception as exc:
            if attempt == ROUTING_MAX_ATTEMPTS or not _recoverable_routing_error(exc):
                raise
            services.emit_progress(
                "routing returned invalid or filtered structured output; "
                f"retrying with exact document identifiers ({attempt + 1}/"
                f"{ROUTING_MAX_ATTEMPTS})"
            )
            current_request = request.model_copy(
                update={
                    "prompt": _routing_recovery_prompt(
                        request.prompt,
                        known_document_ids=known_document_ids,
                    )
                }
            )
            continue
        return (
            generated,
            selected_document_ids,
            response.metadata,
            _combine_usage_records(usage_records),
        )

    raise AssertionError("routing recovery loop terminated unexpectedly")


def _routing_recovery_prompt(prompt: str, *, known_document_ids: set[str]) -> str:
    return (
        f"{sanitize_for_model_recovery_prompt(prompt)}\n\n"
        "Validation correction: return one complete structured routing decision. "
        "selected_document_ids may contain only these exact identifiers: "
        f"{json.dumps(sorted(known_document_ids))}. Do not invent or alter an identifier."
    )


def _recoverable_routing_error(exc: Exception) -> bool:
    if isinstance(exc, ValidationError):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "cannot assist",
            "can't assist",
            "content_filter",
            "invalid json",
            "map router did not select",
            "map router selected unknown",
            "parsed structured output",
            "response ended with status incomplete",
            "unable to assist",
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


def _response_model_id(metadata: dict) -> str | None:
    for name in ("deployment", "model", "client"):
        value = metadata.get(name)
        if value:
            return str(value)
    return None


def _validated_document_ids(
    document_ids: list[str],
    *,
    known_document_ids: set[str],
    top_k: int,
) -> list[str]:
    selected = []
    seen = set()
    for document_id in document_ids:
        normalized = document_id.strip()
        if normalized not in known_document_ids:
            raise ValueError(f"map router selected unknown document identifier: {normalized!r}")
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(normalized)
        if len(selected) >= top_k:
            break
    return selected
