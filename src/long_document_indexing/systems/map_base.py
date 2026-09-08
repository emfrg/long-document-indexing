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
from long_document_indexing.domain.maps import DocumentMap, IndexArtifact, MapEntry, SourceReference
from long_document_indexing.domain.runs import RagRunRecord, UsageRecord
from long_document_indexing.models.base import GenerationRequest, GenerationResponse
from long_document_indexing.models.structured_outputs import StructuredDocumentMap
from long_document_indexing.prompt_safety import (
    PROMPT_SAFETY_POLICY_VERSION,
    apply_prompt_safety_preamble,
    sanitize_for_model_prompt,
    sanitize_prompt_payload,
)
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
        documents = corpus.document_by_id()
        document_maps = [
            normalize_document_map_source_references(
                document_map, documents[document_map.document_id]
            )
            if document_map.document_id in documents
            else document_map
            for document_map in result.document_maps
        ]
        intermediate_maps = {
            map_id: normalize_document_map_source_references(
                document_map, documents[document_map.document_id]
            )
            if document_map.document_id in documents
            else document_map
            for map_id, document_map in result.intermediate_maps.items()
        }

        for document_map in document_maps:
            map_id, path = write_document_map(
                services.artifact_store,
                system_id=self.id,
                corpus_id=corpus.id,
                document_map=document_map,
            )
            document_map_paths[map_id] = str(path)

        intermediate_map_paths: dict[str, str] = {}
        for map_id, document_map in intermediate_maps.items():
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
            "prompt_safety_policy": PROMPT_SAFETY_POLICY_VERSION,
        }
        if intermediate_map_paths:
            build_metadata["intermediate_map_paths"] = intermediate_map_paths
        if result.metadata:
            build_metadata["strategy_metadata"] = result.metadata
        source_reference_normalization = _source_reference_normalization_metadata(
            [*document_maps, *intermediate_maps.values()]
        )
        if source_reference_normalization:
            build_metadata["source_reference_normalization"] = source_reference_normalization
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
        prompt = apply_prompt_safety_preamble(
            render_prompt(
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
        document_map = normalize_document_map_source_references(
            DocumentMap.model_validate_json(response.content),
            document,
        )
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


def normalize_document_map_source_references(
    document_map: DocumentMap,
    document: Document,
) -> DocumentMap:
    """Repair model-produced source references against canonical document segments."""

    valid_segment_ids = {segment.id for segment in document.segments}
    segment_id_aliases = _segment_id_aliases(document)
    summary = _empty_source_reference_normalization_summary()

    if document_map.document_id != document.id:
        summary["map_document_id_rewrites"] += 1

    entries = [
        _normalize_entry_source_references(
            entry,
            document_id=document.id,
            valid_segment_ids=valid_segment_ids,
            segment_id_aliases=segment_id_aliases,
            summary=summary,
        )
        for entry in document_map.entries
    ]
    facets = dict(document_map.facets)
    if _has_source_reference_repairs(summary):
        facets[_SOURCE_REFERENCE_NORMALIZATION_FACET] = _merge_source_reference_summaries(
            facets.get(_SOURCE_REFERENCE_NORMALIZATION_FACET),
            summary,
        )

    return document_map.model_copy(
        update={
            "document_id": document.id,
            "entries": entries,
            "facets": facets,
        }
    )


def _generator_client(services: Services):
    if services.generator_client is None:
        raise RuntimeError(
            "Document-map systems require a generator client. "
            "The default CLI uses FakeTextGenerationClient for local smoke runs."
        )
    return services.generator_client


_SOURCE_REFERENCE_NORMALIZATION_FACET = "source_reference_normalization"
_SOURCE_REFERENCE_COUNT_KEYS = (
    "map_document_id_rewrites",
    "reference_document_id_rewrites",
    "segment_id_alias_rewrites",
    "references_dropped",
    "invalid_segment_ids_dropped",
    "duplicate_segment_ids_dropped",
    "entries_without_source_references",
)
_SOURCE_REFERENCE_SAMPLE_KEYS = (
    "wrong_reference_document_ids",
    "invalid_segment_ids",
    "rewritten_segment_id_aliases",
)
_SOURCE_REFERENCE_SAMPLE_LIMIT = 20


def _normalize_entry_source_references(
    entry: MapEntry,
    *,
    document_id: str,
    valid_segment_ids: set[str],
    segment_id_aliases: dict[str, str],
    summary: dict[str, Any],
) -> MapEntry:
    source_references = [
        reference
        for reference in (
            _normalize_source_reference(
                reference,
                document_id=document_id,
                valid_segment_ids=valid_segment_ids,
                segment_id_aliases=segment_id_aliases,
                summary=summary,
            )
            for reference in entry.source_references
        )
        if reference is not None
    ]
    children = [
        _normalize_entry_source_references(
            child,
            document_id=document_id,
            valid_segment_ids=valid_segment_ids,
            segment_id_aliases=segment_id_aliases,
            summary=summary,
        )
        for child in entry.children
    ]
    if not source_references:
        summary["entries_without_source_references"] += 1

    return entry.model_copy(
        update={
            "source_references": source_references,
            "children": children,
        }
    )


def _normalize_source_reference(
    reference: SourceReference,
    *,
    document_id: str,
    valid_segment_ids: set[str],
    segment_id_aliases: dict[str, str],
    summary: dict[str, Any],
) -> SourceReference | None:
    segment_ids: list[str] = []
    seen_segment_ids = set()
    for segment_id in reference.segment_ids:
        canonical_segment_id = segment_id_aliases.get(segment_id)
        if canonical_segment_id is None or canonical_segment_id not in valid_segment_ids:
            summary["invalid_segment_ids_dropped"] += 1
            _append_source_reference_sample(summary, "invalid_segment_ids", segment_id)
            continue
        if canonical_segment_id != segment_id:
            summary["segment_id_alias_rewrites"] += 1
            _append_source_reference_sample(
                summary,
                "rewritten_segment_id_aliases",
                f"{segment_id}->{canonical_segment_id}",
            )
        if canonical_segment_id in seen_segment_ids:
            summary["duplicate_segment_ids_dropped"] += 1
            continue
        segment_ids.append(canonical_segment_id)
        seen_segment_ids.add(canonical_segment_id)

    if not segment_ids:
        summary["references_dropped"] += 1
        if reference.document_id != document_id:
            _append_source_reference_sample(
                summary,
                "wrong_reference_document_ids",
                reference.document_id,
            )
        return None

    if reference.document_id != document_id:
        summary["reference_document_id_rewrites"] += 1
        _append_source_reference_sample(
            summary,
            "wrong_reference_document_ids",
            reference.document_id,
        )

    return SourceReference(document_id=document_id, segment_ids=segment_ids)


def _empty_source_reference_normalization_summary() -> dict[str, Any]:
    return {
        **dict.fromkeys(_SOURCE_REFERENCE_COUNT_KEYS, 0),
        **{key: [] for key in _SOURCE_REFERENCE_SAMPLE_KEYS},
    }


def _segment_id_aliases(document: Document) -> dict[str, str]:
    aliases: dict[str, set[str]] = {}
    for segment in document.segments:
        for alias in _candidate_segment_id_aliases(segment.id, document.id):
            aliases.setdefault(alias, set()).add(segment.id)
    return {
        alias: next(iter(segment_ids))
        for alias, segment_ids in aliases.items()
        if len(segment_ids) == 1
    }


def _candidate_segment_id_aliases(segment_id: str, document_id: str) -> set[str]:
    aliases = {segment_id}
    if segment_id.startswith(f"{document_id}:"):
        aliases.add(segment_id.removeprefix(f"{document_id}:"))
    parts = segment_id.split(":")
    aliases.update(":".join(parts[index:]) for index in range(1, len(parts)))
    return aliases


def _has_source_reference_repairs(summary: dict[str, Any]) -> bool:
    return any(int(summary[key]) > 0 for key in _SOURCE_REFERENCE_COUNT_KEYS)


def _append_source_reference_sample(
    summary: dict[str, Any],
    key: str,
    value: str,
) -> None:
    sample = summary[key]
    if value and value not in sample and len(sample) < _SOURCE_REFERENCE_SAMPLE_LIMIT:
        sample.append(value)


def _merge_source_reference_summaries(
    existing: Any,
    new: dict[str, Any],
) -> dict[str, Any]:
    merged = _empty_source_reference_normalization_summary()
    if isinstance(existing, dict):
        for key in _SOURCE_REFERENCE_COUNT_KEYS:
            merged[key] = int(existing.get(key, 0))
        for key in _SOURCE_REFERENCE_SAMPLE_KEYS:
            values = existing.get(key, [])
            if isinstance(values, list):
                for value in values:
                    _append_source_reference_sample(merged, key, str(value))

    for key in _SOURCE_REFERENCE_COUNT_KEYS:
        merged[key] += int(new[key])
    for key in _SOURCE_REFERENCE_SAMPLE_KEYS:
        for value in new[key]:
            _append_source_reference_sample(merged, key, str(value))
    return merged


def _source_reference_normalization_metadata(
    document_maps: list[DocumentMap],
) -> dict[str, Any]:
    merged = _empty_source_reference_normalization_summary()
    repaired_map_count = 0
    for document_map in document_maps:
        summary = document_map.facets.get(_SOURCE_REFERENCE_NORMALIZATION_FACET)
        if not isinstance(summary, dict):
            continue
        repaired_map_count += 1
        merged = _merge_source_reference_summaries(merged, summary)

    if repaired_map_count == 0 or not _has_source_reference_repairs(merged):
        return {}
    return {"repaired_map_count": repaired_map_count, **merged}


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
        f"[{segment.id}]\n{sanitize_for_model_prompt(segment.text)}"
        for segment in sorted(segments, key=lambda item: item.order)
    )


def _prompt_values(
    base: dict[str, Any],
    extra_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    values = {key: sanitize_prompt_payload(value, key=key) for key, value in base.items()}
    for key, value in (extra_metadata or {}).items():
        safe_value = sanitize_prompt_payload(value, key=key)
        if isinstance(safe_value, str | int | float | bool) or safe_value is None:
            values[key] = "" if safe_value is None else safe_value
        else:
            values[key] = json.dumps(safe_value, indent=2, sort_keys=True)
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
    prompt = apply_prompt_safety_preamble(
        render_prompt(
            template,
            {
                "document_id": document.id,
                "title": sanitize_for_model_prompt(document.title or document.id),
                "construction_method": strategy,
                "partial_map_count": len(partial_maps),
                "partial_maps": json.dumps(
                    sanitize_prompt_payload(
                        [document_map.model_dump(mode="json") for document_map in partial_maps]
                    ),
                    indent=2,
                    sort_keys=True,
                ),
            },
        )
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
    return (
        normalize_document_map_source_references(
            DocumentMap.model_validate_json(response.content),
            document,
        ),
        response.usage,
    )


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
    prompt = apply_prompt_safety_preamble(
        render_prompt(
            template,
            {
                "document_id": document.id,
                "title": sanitize_for_model_prompt(document.title or document.id),
                "construction_method": strategy,
                "segment_id": segment.id,
                "segment_text": sanitize_for_model_prompt(segment.text),
                "existing_map": json.dumps(
                    sanitize_prompt_payload(
                        existing_map.model_dump(mode="json") if existing_map is not None else {}
                    ),
                    indent=2,
                    sort_keys=True,
                ),
            },
        )
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
    return (
        normalize_document_map_source_references(
            DocumentMap.model_validate_json(response.content),
            document,
        ),
        response.usage,
    )
