from __future__ import annotations

from long_document_indexing.config import SharedPipelineConfig
from long_document_indexing.domain.corpus import Corpus, Document, Segment
from long_document_indexing.domain.maps import DocumentMap, MapEntry
from long_document_indexing.domain.runs import UsageRecord
from long_document_indexing.prompt_safety import PROMPT_SAFETY_POLICY_VERSION
from long_document_indexing.services import Services
from long_document_indexing.storage.map_checkpoints import (
    input_signature,
    read_map_checkpoint,
    write_map_checkpoint,
    write_map_checkpoint_failure,
)
from long_document_indexing.systems.map_base import (
    DocumentMapSystemBase,
    MapBuildResult,
    combine_usage,
    reduce_maps_with_model,
)
from long_document_indexing.telemetry.tracing import stable_id


class OutlineThenFillSystem(DocumentMapSystemBase):
    """Plan a document outline, then fill outline nodes from source segments."""

    id = "outline_then_fill"
    construction_method = "outline_then_fill"

    def __init__(self, *, outline_depth: int = 2, outline_max_nodes: int = 12) -> None:
        if outline_depth < 1:
            raise ValueError("outline_depth must be positive")
        if outline_max_nodes < 1:
            raise ValueError("outline_max_nodes must be positive")
        self.outline_depth = outline_depth
        self.outline_max_nodes = outline_max_nodes

    async def build_document_maps(
        self,
        corpus: Corpus,
        services: Services,
        pipeline: SharedPipelineConfig,
    ) -> MapBuildResult:
        del pipeline
        document_maps = []
        usage_records = []
        statuses = {}
        intermediate_maps = {}
        documents_metadata = {}
        checkpoint_stats = _checkpoint_stats()

        for document_index, document in enumerate(corpus.documents, start=1):
            services.emit_progress(
                "outline_then_fill indexing "
                f"{corpus.id}: document {document_index}/{len(corpus.documents)} "
                f"{document.id} ({len(document.segments)} segment(s))"
            )
            outline_map, outline_usage = await self._build_outline_map(
                corpus=corpus,
                document=document,
                services=services,
                checkpoint_stats=checkpoint_stats,
            )
            usage_records.append(outline_usage)
            outline_nodes = _outline_nodes(outline_map, self.outline_max_nodes)
            segment_groups = _assign_segments_to_outline_nodes(
                document.segments,
                len(outline_nodes),
            )
            intermediate_maps[
                _intermediate_map_id(self.id, corpus.id, document.id, "outline", 0)
            ] = _annotated_map(
                outline_map,
                construction_method=self.construction_method,
                phase="outline",
                index=0,
                extra_facets={
                    "outline_depth": self.outline_depth,
                    "outline_node_count": len(outline_nodes),
                },
            )

            fill_maps = []
            for index, segments in enumerate(segment_groups):
                outline_node = outline_nodes[index] if index < len(outline_nodes) else None
                fill_map, fill_usage = await self._build_fill_map(
                    corpus=corpus,
                    document=document,
                    segments=segments,
                    outline_node=outline_node,
                    index=index,
                    group_count=len(segment_groups),
                    services=services,
                    checkpoint_stats=checkpoint_stats,
                )
                usage_records.append(fill_usage)
                annotated_fill_map = _annotated_map(
                    fill_map,
                    construction_method=self.construction_method,
                    phase="fill",
                    index=index,
                    extra_facets={
                        "outline_depth": self.outline_depth,
                        "outline_node_count": len(outline_nodes),
                    },
                )
                fill_maps.append(annotated_fill_map)
                intermediate_maps[
                    _intermediate_map_id(self.id, corpus.id, document.id, "fill", index)
                ] = annotated_fill_map

            final_map, assemble_usage = await self._assemble_document_map(
                corpus=corpus,
                document=document,
                fill_maps=fill_maps,
                services=services,
                checkpoint_stats=checkpoint_stats,
            )
            usage_records.extend(assemble_usage)
            document_maps.append(
                _annotated_map(
                    final_map,
                    construction_method=self.construction_method,
                    phase="final",
                    index=0,
                    extra_facets={
                        "outline_depth": self.outline_depth,
                        "outline_node_count": len(outline_nodes),
                        "fill_map_count": len(fill_maps),
                    },
                )
            )
            statuses[document.id] = "indexed"
            documents_metadata[document.id] = {
                "outline_node_count": len(outline_nodes),
                "fill_map_count": len(fill_maps),
            }
            services.emit_progress(f"outline_then_fill indexed {corpus.id}/{document.id}")

        return MapBuildResult(
            document_maps=document_maps,
            usage=combine_usage(*usage_records),
            statuses=statuses,
            intermediate_maps=intermediate_maps,
            metadata={
                "outline_depth": self.outline_depth,
                "outline_max_nodes": self.outline_max_nodes,
                "documents": documents_metadata,
                "checkpointing": checkpoint_stats,
            },
        )

    async def _build_outline_map(
        self,
        *,
        corpus: Corpus,
        document: Document,
        services: Services,
        checkpoint_stats: dict[str, int],
    ) -> tuple[DocumentMap, UsageRecord]:
        phase = "outline"
        metadata = {
            "phase": "outline",
            "outline_depth": self.outline_depth,
            "outline_max_nodes": self.outline_max_nodes,
            "segment_count": len(document.segments),
        }
        signature = input_signature(
            self.id,
            PROMPT_SAFETY_POLICY_VERSION,
            corpus.id,
            document.id,
            self.construction_method,
            phase,
            self.outline_depth,
            self.outline_max_nodes,
            [_segment_signature_payload(segment) for segment in document.segments],
        )
        checkpoint = (
            read_map_checkpoint(
                services.artifact_store,
                system_id=self.id,
                corpus_id=corpus.id,
                document_id=document.id,
                phase=phase,
                input_signature=signature,
            )
            if services.resume_checkpoints
            else None
        )
        if checkpoint is not None:
            checkpoint_stats["outline_reused"] += 1
            services.emit_progress(
                f"outline_then_fill reused outline checkpoint {corpus.id}/{document.id}"
            )
            return checkpoint.document_map, checkpoint.usage

        try:
            outline_map, usage = await self.generate_document_map(
                document=document,
                segments=document.segments,
                services=services,
                prompt_parts=("outline_then_fill", "outline"),
                extra_metadata=metadata,
            )
        except Exception as exc:
            write_map_checkpoint_failure(
                services.artifact_store,
                system_id=self.id,
                corpus_id=corpus.id,
                document_id=document.id,
                phase=phase,
                input_signature=signature,
                error=exc,
                metadata=metadata,
            )
            services.emit_progress(
                "outline_then_fill failed outline "
                f"{corpus.id}/{document.id}: {exc.__class__.__name__}: {exc}"
            )
            raise

        write_map_checkpoint(
            services.artifact_store,
            system_id=self.id,
            corpus_id=corpus.id,
            document_id=document.id,
            phase=phase,
            input_signature=signature,
            document_map=outline_map,
            usage=usage,
            metadata=metadata,
        )
        checkpoint_stats["outline_written"] += 1
        services.emit_progress(
            "outline_then_fill wrote outline checkpoint "
            f"{corpus.id}/{document.id} "
            f"({usage.model_calls} call(s), "
            f"{usage.input_tokens + usage.output_tokens} token(s))"
        )
        return outline_map, usage

    async def _build_fill_map(
        self,
        *,
        corpus: Corpus,
        document: Document,
        segments: list[Segment],
        outline_node: MapEntry | None,
        index: int,
        group_count: int,
        services: Services,
        checkpoint_stats: dict[str, int],
    ) -> tuple[DocumentMap, UsageRecord]:
        phase = f"fill-{index + 1:04d}"
        outline_node_payload = (
            outline_node.model_dump(mode="json") if outline_node is not None else None
        )
        metadata = {
            "phase": "fill",
            "outline_depth": self.outline_depth,
            "outline_node": outline_node_payload,
            "fill_index": index,
            "fill_count": group_count,
            "source_segment_ids": [segment.id for segment in segments],
        }
        signature = input_signature(
            self.id,
            PROMPT_SAFETY_POLICY_VERSION,
            corpus.id,
            document.id,
            self.construction_method,
            phase,
            self.outline_depth,
            self.outline_max_nodes,
            outline_node_payload,
            [_segment_signature_payload(segment) for segment in segments],
        )
        checkpoint = (
            read_map_checkpoint(
                services.artifact_store,
                system_id=self.id,
                corpus_id=corpus.id,
                document_id=document.id,
                phase=phase,
                input_signature=signature,
            )
            if services.resume_checkpoints
            else None
        )
        if checkpoint is not None:
            checkpoint_stats["fill_reused"] += 1
            services.emit_progress(
                "outline_then_fill reused fill checkpoint "
                f"{corpus.id}/{document.id} group {index + 1}/{group_count}"
            )
            return checkpoint.document_map, checkpoint.usage

        try:
            fill_map, usage = await self.generate_document_map(
                document=document,
                segments=segments,
                services=services,
                prompt_parts=("outline_then_fill", "fill"),
                extra_metadata=metadata,
            )
        except Exception as exc:
            write_map_checkpoint_failure(
                services.artifact_store,
                system_id=self.id,
                corpus_id=corpus.id,
                document_id=document.id,
                phase=phase,
                input_signature=signature,
                error=exc,
                metadata=metadata,
            )
            services.emit_progress(
                "outline_then_fill failed fill "
                f"{corpus.id}/{document.id} group {index + 1}/{group_count}: "
                f"{exc.__class__.__name__}: {exc}"
            )
            raise

        write_map_checkpoint(
            services.artifact_store,
            system_id=self.id,
            corpus_id=corpus.id,
            document_id=document.id,
            phase=phase,
            input_signature=signature,
            document_map=fill_map,
            usage=usage,
            metadata=metadata,
        )
        checkpoint_stats["fill_written"] += 1
        services.emit_progress(
            "outline_then_fill wrote fill checkpoint "
            f"{corpus.id}/{document.id} group {index + 1}/{group_count} "
            f"({usage.model_calls} call(s), "
            f"{usage.input_tokens + usage.output_tokens} token(s))"
        )
        return fill_map, usage

    async def _assemble_document_map(
        self,
        *,
        corpus: Corpus,
        document: Document,
        fill_maps: list[DocumentMap],
        services: Services,
        checkpoint_stats: dict[str, int],
    ) -> tuple[DocumentMap, list[UsageRecord]]:
        if len(fill_maps) == 1:
            return fill_maps[0], []

        phase = "assemble"
        metadata = {"phase": "assemble", "fill_map_count": len(fill_maps)}
        signature = input_signature(
            self.id,
            PROMPT_SAFETY_POLICY_VERSION,
            corpus.id,
            document.id,
            self.construction_method,
            phase,
            self.outline_depth,
            self.outline_max_nodes,
            [document_map.model_dump_json() for document_map in fill_maps],
        )
        checkpoint = (
            read_map_checkpoint(
                services.artifact_store,
                system_id=self.id,
                corpus_id=corpus.id,
                document_id=document.id,
                phase=phase,
                input_signature=signature,
            )
            if services.resume_checkpoints
            else None
        )
        if checkpoint is not None:
            checkpoint_stats["assemble_reused"] += 1
            services.emit_progress(
                f"outline_then_fill reused assemble checkpoint {corpus.id}/{document.id}"
            )
            return checkpoint.document_map, [checkpoint.usage]

        try:
            final_map, usage = await reduce_maps_with_model(
                document=document,
                partial_maps=fill_maps,
                services=services,
                strategy=self.construction_method,
                prompt_parts=("outline_then_fill", "assemble"),
            )
        except Exception as exc:
            write_map_checkpoint_failure(
                services.artifact_store,
                system_id=self.id,
                corpus_id=corpus.id,
                document_id=document.id,
                phase=phase,
                input_signature=signature,
                error=exc,
                metadata=metadata,
            )
            services.emit_progress(
                "outline_then_fill failed assemble "
                f"{corpus.id}/{document.id}: {exc.__class__.__name__}: {exc}"
            )
            raise

        write_map_checkpoint(
            services.artifact_store,
            system_id=self.id,
            corpus_id=corpus.id,
            document_id=document.id,
            phase=phase,
            input_signature=signature,
            document_map=final_map,
            usage=usage,
            metadata=metadata,
        )
        checkpoint_stats["assemble_written"] += 1
        services.emit_progress(
            "outline_then_fill wrote assemble checkpoint "
            f"{corpus.id}/{document.id} "
            f"({usage.model_calls} call(s), "
            f"{usage.input_tokens + usage.output_tokens} token(s))"
        )
        return final_map, [usage]


def _outline_nodes(outline_map: DocumentMap, max_nodes: int) -> list[MapEntry]:
    nodes = [entry for root in outline_map.entries for entry in root.walk()]
    return nodes[:max_nodes]


def _assign_segments_to_outline_nodes(
    segments: list[Segment],
    node_count: int,
) -> list[list[Segment]]:
    ordered = sorted(segments, key=lambda item: item.order)
    if not ordered:
        return []
    group_count = max(1, node_count)
    groups = [[] for _ in range(group_count)]
    for index, segment in enumerate(ordered):
        groups[index % group_count].append(segment)
    return [group for group in groups if group]


def _annotated_map(
    document_map: DocumentMap,
    *,
    construction_method: str,
    phase: str,
    index: int,
    extra_facets: dict[str, int] | None = None,
) -> DocumentMap:
    return document_map.model_copy(
        update={
            "construction_method": construction_method,
            "facets": {
                **document_map.facets,
                "outline_phase": phase,
                "outline_index": index,
                **(extra_facets or {}),
            },
        }
    )


def _intermediate_map_id(
    system_id: str,
    corpus_id: str,
    document_id: str,
    phase: str,
    index: int,
) -> str:
    return stable_id("map", system_id, corpus_id, document_id, phase, index)


def _segment_signature_payload(segment: Segment) -> dict[str, object]:
    return {"id": segment.id, "order": segment.order, "text": segment.text}


def _checkpoint_stats() -> dict[str, int]:
    return {
        "outline_reused": 0,
        "outline_written": 0,
        "fill_reused": 0,
        "fill_written": 0,
        "assemble_reused": 0,
        "assemble_written": 0,
    }
