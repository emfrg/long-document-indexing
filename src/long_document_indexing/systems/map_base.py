from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from long_document_indexing.answering import answer_from_retrieved_evidence, query_usage
from long_document_indexing.config import SharedPipelineConfig
from long_document_indexing.domain.benchmark import BenchmarkItem
from long_document_indexing.domain.corpus import Corpus, Document, Segment
from long_document_indexing.domain.maps import DocumentMap, IndexArtifact
from long_document_indexing.domain.runs import RagRunRecord, UsageRecord
from long_document_indexing.models.base import GenerationRequest, GenerationResponse
from long_document_indexing.models.structured_outputs import StructuredDocumentMap
from long_document_indexing.prompts import render_prompt
from long_document_indexing.services import Services
from long_document_indexing.storage.maps import read_document_map, write_document_map
from long_document_indexing.telemetry.tracing import stable_id, stable_query_run_id
from long_document_indexing.text import (
    approximate_token_count,
    lexical_similarity,
)


@dataclass
class MapBuildResult:
    document_maps: list[DocumentMap]
    usage: UsageRecord = field(default_factory=UsageRecord)
    statuses: dict[str, str] = field(default_factory=dict)
    intermediate_maps: dict[str, DocumentMap] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentMapSystemBase(ABC):
    """Shared index/query behavior for systems that construct `DocumentMap` artifacts."""

    id: str
    construction_method: str

    async def build_index(
        self,
        corpus: Corpus,
        services: Services,
        pipeline: SharedPipelineConfig,
    ) -> IndexArtifact:
        retrieval_index_id = await services.retrieval_backend.index(corpus)
        result = await self.build_document_maps(corpus, services, pipeline)

        document_map_paths: dict[str, str] = {}
        for document_map in result.document_maps:
            map_id, path = write_document_map(
                services.artifact_store,
                system_id=self.id,
                corpus_id=corpus.id,
                document_map=document_map,
            )
            document_map_paths[map_id] = str(path)

        intermediate_map_paths: dict[str, str] = {}
        for map_id, document_map in result.intermediate_maps.items():
            path = services.artifact_store.write_json(
                f"indexes/{self.id}/intermediate_maps/{map_id}.json",
                document_map,
            )
            intermediate_map_paths[map_id] = str(path)

        artifact_id = stable_id("index", self.id, corpus.id, retrieval_index_id)
        build_metadata: dict[str, Any] = {
            "retrieval_index_id": retrieval_index_id,
            "retrieval_backend": services.retrieval_backend.__class__.__name__,
            "document_map_paths": document_map_paths,
            "document_statuses": result.statuses,
            "usage": result.usage.model_dump(mode="json"),
        }
        if intermediate_map_paths:
            build_metadata["intermediate_map_paths"] = intermediate_map_paths
        if result.metadata:
            build_metadata["strategy_metadata"] = result.metadata
        artifact = IndexArtifact(
            id=artifact_id,
            system_id=self.id,
            corpus_id=corpus.id,
            artifact_path=str(
                services.artifact_store.path(f"indexes/{self.id}/{artifact_id}.json")
            ),
            document_map_ids=list(document_map_paths),
            build_metadata=build_metadata,
        )
        services.artifact_store.write_json(f"indexes/{self.id}/{artifact_id}.json", artifact)
        return artifact

    async def run_query(
        self,
        item: BenchmarkItem,
        corpus: Corpus,
        index_artifact: IndexArtifact,
        services: Services,
        pipeline: SharedPipelineConfig,
        *,
        experiment_id: str,
        repetition: int,
    ) -> RagRunRecord:
        started = time.perf_counter()
        document_maps = _load_document_maps(index_artifact)
        selected_document_ids = _route_documents(
            item.query,
            document_maps,
            corpus,
            top_k=pipeline.selected_documents,
        )
        retrieval_index_id = _retrieval_index_id(index_artifact)
        retrieved_items = await services.retrieval_backend.search(
            retrieval_index_id,
            item.query,
            document_ids=selected_document_ids,
            top_k=pipeline.retrieved_segments,
        )
        answer_result = await answer_from_retrieved_evidence(
            item=item,
            retrieved_items=retrieved_items,
            services=services,
            extractive_prefix="Local mapped-system answer from retrieved evidence",
        )
        duration_ms = (time.perf_counter() - started) * 1000.0
        return RagRunRecord(
            run_id=stable_query_run_id(experiment_id, self.id, item.id, repetition),
            experiment_id=experiment_id,
            system_id=self.id,
            corpus_id=item.corpus_id,
            item_id=item.id,
            repetition=repetition,
            selected_document_ids=selected_document_ids,
            retrieved_items=retrieved_items,
            answer=answer_result.answer,
            citations=answer_result.citations,
            usage=query_usage(
                answer_result.usage,
                tool_calls=2,
                duration_ms=duration_ms,
            ),
            status="succeeded",
        )

    @abstractmethod
    async def build_document_maps(
        self,
        corpus: Corpus,
        services: Services,
        pipeline: SharedPipelineConfig,
    ) -> MapBuildResult:
        """Build strategy-specific maps for every eligible document in a corpus."""

    async def generate_document_map(
        self,
        *,
        document: Document,
        segments: list[Segment],
        services: Services,
        prompt_parts: tuple[str, ...],
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[DocumentMap, UsageRecord]:
        client = _generator_client(services)
        template = services.prompt_loader.load(*prompt_parts)
        prompt = render_prompt(
            template,
            _prompt_values(
                {
                    "document_id": document.id,
                    "title": document.title or document.id,
                    "construction_method": self.construction_method,
                    "segment_count": len(segments),
                    "segments": _segments_for_prompt(segments),
                },
                extra_metadata,
            ),
        )
        response = await client.generate(
            GenerationRequest(
                prompt=prompt,
                prompt_name="/".join(prompt_parts),
                metadata={
                    "task": "document_map",
                    "strategy": self.construction_method,
                    "document_id": document.id,
                    "segments": [_segment_payload(segment) for segment in segments],
                    **(extra_metadata or {}),
                },
                response_model=StructuredDocumentMap,
            )
        )
        document_map = DocumentMap.model_validate_json(response.content)
        return document_map, response.usage


def combine_usage(*records: UsageRecord) -> UsageRecord:
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


def _generator_client(services: Services):
    if services.generator_client is None:
        raise RuntimeError(
            "Document-map systems require a generator client. "
            "The default CLI uses FakeTextGenerationClient for local smoke runs."
        )
    return services.generator_client


def _segment_payload(segment: Segment) -> dict[str, Any]:
    return {
        "id": segment.id,
        "document_id": segment.document_id,
        "order": segment.order,
        "text": segment.text,
        "metadata": segment.metadata,
    }


def _segments_for_prompt(segments: list[Segment]) -> str:
    return "\n\n".join(
        f"[{segment.id}]\n{segment.text}"
        for segment in sorted(segments, key=lambda item: item.order)
    )


def _prompt_values(
    base: dict[str, Any],
    extra_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    values = dict(base)
    for key, value in (extra_metadata or {}).items():
        if isinstance(value, str | int | float | bool) or value is None:
            values[key] = "" if value is None else value
        else:
            values[key] = json.dumps(value, indent=2, sort_keys=True)
    return values


def _load_document_maps(index_artifact: IndexArtifact) -> list[DocumentMap]:
    paths = index_artifact.build_metadata.get("document_map_paths", {})
    if not isinstance(paths, dict):
        raise ValueError("index artifact build_metadata.document_map_paths must be a mapping")
    return [read_document_map(path) for path in paths.values()]


def _route_documents(
    query: str,
    document_maps: list[DocumentMap],
    corpus: Corpus,
    *,
    top_k: int,
) -> list[str]:
    if top_k < 1:
        return []
    if not document_maps:
        return [document.id for document in corpus.documents[:top_k]]

    scored = [
        (
            lexical_similarity(query, _map_text(document_map)),
            document_map.document_id,
        )
        for document_map in document_maps
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [document_id for _, document_id in scored[:top_k]]


def _map_text(document_map: DocumentMap) -> str:
    parts = [document_map.overview]
    for root in document_map.entries:
        for entry in root.walk():
            parts.extend([entry.kind, entry.label, entry.summary])
    return "\n".join(parts)


def _retrieval_index_id(index_artifact: IndexArtifact) -> str:
    retrieval_index_id = index_artifact.build_metadata.get("retrieval_index_id")
    if not isinstance(retrieval_index_id, str) or not retrieval_index_id.strip():
        raise ValueError("mapped system index artifact is missing retrieval_index_id")
    return retrieval_index_id


def document_source_token_count(document: Document) -> int:
    return sum(approximate_token_count(segment.text) for segment in document.segments)


async def reduce_maps_with_model(
    *,
    document: Document,
    partial_maps: list[DocumentMap],
    services: Services,
    strategy: str,
    prompt_parts: tuple[str, ...],
) -> tuple[DocumentMap, UsageRecord]:
    client = _generator_client(services)
    template = services.prompt_loader.load(*prompt_parts)
    prompt = render_prompt(
        template,
        {
            "document_id": document.id,
            "title": document.title or document.id,
            "construction_method": strategy,
            "partial_map_count": len(partial_maps),
            "partial_maps": json.dumps(
                [document_map.model_dump(mode="json") for document_map in partial_maps],
                indent=2,
                sort_keys=True,
            ),
        },
    )
    response: GenerationResponse = await client.generate(
        GenerationRequest(
            prompt=prompt,
            prompt_name="/".join(prompt_parts),
            metadata={
                "task": "reduce_document_maps",
                "strategy": strategy,
                "document_id": document.id,
                "partial_maps": [
                    document_map.model_dump(mode="json") for document_map in partial_maps
                ],
            },
            response_model=StructuredDocumentMap,
        )
    )
    return DocumentMap.model_validate_json(response.content), response.usage


async def refine_map_with_model(
    *,
    document: Document,
    segment: Segment,
    existing_map: DocumentMap | None,
    services: Services,
    strategy: str,
    prompt_parts: tuple[str, ...],
) -> tuple[DocumentMap, UsageRecord]:
    client = _generator_client(services)
    template = services.prompt_loader.load(*prompt_parts)
    prompt = render_prompt(
        template,
        {
            "document_id": document.id,
            "title": document.title or document.id,
            "construction_method": strategy,
            "segment_id": segment.id,
            "segment_text": segment.text,
            "existing_map": json.dumps(
                existing_map.model_dump(mode="json") if existing_map is not None else {},
                indent=2,
                sort_keys=True,
            ),
        },
    )
    response = await client.generate(
        GenerationRequest(
            prompt=prompt,
            prompt_name="/".join(prompt_parts),
            metadata={
                "task": "refine_document_map",
                "strategy": strategy,
                "document_id": document.id,
                "segments": [_segment_payload(segment)],
                "existing_map": existing_map.model_dump(mode="json")
                if existing_map is not None
                else None,
            },
            response_model=StructuredDocumentMap,
        )
    )
    return DocumentMap.model_validate_json(response.content), response.usage
