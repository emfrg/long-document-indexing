from __future__ import annotations

import json
from dataclasses import dataclass

from long_document_indexing.domain.maps import DocumentMap
from long_document_indexing.domain.runs import RoutingDecision, UsageRecord
from long_document_indexing.models.base import GenerationRequest
from long_document_indexing.models.structured_outputs import StructuredRoutingDecision
from long_document_indexing.progress import await_with_progress
from long_document_indexing.prompt_safety import (
    apply_prompt_safety_preamble,
    sanitize_for_model_prompt,
    sanitize_prompt_payload,
)
from long_document_indexing.prompts import render_prompt
from long_document_indexing.services import Services

ROUTING_POLICY_VERSION = "model-map-router/v1"


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
    response = await await_with_progress(
        client.generate(
            GenerationRequest(
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
        ),
        emit=services.progress,
        message="routing waiting for model response",
    )
    generated = StructuredRoutingDecision.model_validate_json(response.content)
    known_document_ids = {document_map.document_id for document_map in document_maps}
    selected_document_ids = _validated_document_ids(
        generated.selected_document_ids,
        known_document_ids=known_document_ids,
        top_k=top_k,
    )
    if not selected_document_ids:
        raise ValueError("map router did not select any known document identifiers")

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
            model_id=_response_model_id(response.metadata),
        ),
        usage=response.usage,
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
