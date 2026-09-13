from __future__ import annotations

from long_document_indexing.config import SharedPipelineConfig
from long_document_indexing.domain.corpus import Corpus, Document, Segment
from long_document_indexing.domain.maps import DocumentMap
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


class HierarchicalMapSystem(DocumentMapSystemBase):
    """Build leaf maps, then reduce them into a bounded hierarchy."""

    id = "hierarchical_map"
    construction_method = "hierarchical_map"

    def __init__(self, *, branching_factor: int = 4, max_levels: int = 3) -> None:
        if branching_factor < 2:
            raise ValueError("branching_factor must be at least 2")
        if max_levels < 1:
            raise ValueError("max_levels must be positive")
        self.branching_factor = branching_factor
        self.max_levels = max_levels

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
                "hierarchical_map indexing "
                f"{corpus.id}: document {document_index}/{len(corpus.documents)} "
                f"{document.id} ({len(document.segments)} segment(s))"
            )
            leaf_maps, leaf_usage = await self._build_leaf_maps(
                corpus=corpus,
                document=document,
                services=services,
                checkpoint_stats=checkpoint_stats,
            )
            usage_records.extend(leaf_usage)
            for index, document_map in enumerate(leaf_maps):
                intermediate_maps[
                    _intermediate_map_id(self.id, corpus.id, document.id, "leaf", 0, index)
                ] = _annotated_map(
                    document_map,
                    construction_method=self.construction_method,
                    role="leaf",
                    level=0,
                    index=index,
                )

            (
                final_map,
                reduction_usage,
                hierarchy_metadata,
                reduced_intermediate_maps,
            ) = await self._reduce_hierarchy(
                corpus=corpus,
                document=document,
                leaf_maps=leaf_maps,
                services=services,
                checkpoint_stats=checkpoint_stats,
            )
            usage_records.extend(reduction_usage)
            intermediate_maps.update(reduced_intermediate_maps)

            if final_map is None:
                statuses[document.id] = "empty_document"
                documents_metadata[document.id] = {"leaf_map_count": 0, "levels": 0}
                continue

            document_maps.append(
                _annotated_map(
                    final_map,
                    construction_method=self.construction_method,
                    role="document",
                    level=hierarchy_metadata["levels"],
                    index=0,
                    extra_facets={
                        "branching_factor": self.branching_factor,
                        "max_levels": self.max_levels,
                        **hierarchy_metadata,
                    },
                )
            )
            statuses[document.id] = "indexed"
            documents_metadata[document.id] = hierarchy_metadata
            services.emit_progress(f"hierarchical_map indexed {corpus.id}/{document.id}")

        return MapBuildResult(
            document_maps=document_maps,
            usage=combine_usage(*usage_records),
            statuses=statuses,
            intermediate_maps=intermediate_maps,
            metadata={
                "branching_factor": self.branching_factor,
                "max_levels": self.max_levels,
                "documents": documents_metadata,
                "checkpointing": checkpoint_stats,
            },
        )

    async def _build_leaf_maps(
        self,
        *,
        corpus: Corpus,
        document: Document,
        services: Services,
        checkpoint_stats: dict[str, int],
    ) -> tuple[list[DocumentMap], list[UsageRecord]]:
        leaf_maps = []
        usage_records = []
        groups = _segment_groups(document, self.branching_factor)
        for index, segments in enumerate(groups):
            phase = f"leaf-{index + 1:04d}"
            metadata = {
                "hierarchy_level": 0,
                "hierarchy_role": "leaf",
                "group_index": index,
                "group_count": len(groups),
                "source_segment_ids": [segment.id for segment in segments],
            }
            signature = input_signature(
                self.id,
                PROMPT_SAFETY_POLICY_VERSION,
                services.map_generation_signature,
                corpus.id,
                document.id,
                self.construction_method,
                phase,
                self.branching_factor,
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
                document_map = checkpoint.document_map
                usage = checkpoint.usage
                checkpoint_stats["leaf_reused"] += 1
                services.emit_progress(
                    "hierarchical_map reused leaf checkpoint "
                    f"{corpus.id}/{document.id} group {index + 1}/{len(groups)}"
                )
            else:
                try:
                    document_map, usage = await self.generate_document_map(
                        document=document,
                        segments=segments,
                        services=services,
                        prompt_parts=("hierarchical_map", "leaf"),
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
                        "hierarchical_map failed leaf "
                        f"{corpus.id}/{document.id} group {index + 1}/{len(groups)}: "
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
                    document_map=document_map,
                    usage=usage,
                    metadata=metadata,
                )
                checkpoint_stats["leaf_written"] += 1
                services.emit_progress(
                    "hierarchical_map wrote leaf checkpoint "
                    f"{corpus.id}/{document.id} group {index + 1}/{len(groups)} "
                    f"({usage.model_calls} call(s), "
                    f"{usage.input_tokens + usage.output_tokens} token(s))"
                )

            leaf_maps.append(document_map)
            usage_records.append(usage)
        return leaf_maps, usage_records

    async def _reduce_hierarchy(
        self,
        *,
        corpus: Corpus,
        document: Document,
        leaf_maps: list[DocumentMap],
        services: Services,
        checkpoint_stats: dict[str, int],
    ) -> tuple[DocumentMap | None, list[UsageRecord], dict[str, int], dict[str, DocumentMap]]:
        if not leaf_maps:
            return None, [], {"leaf_map_count": 0, "reduction_count": 0, "levels": 0}, {}

        current = leaf_maps
        level = 0
        reduction_count = 0
        usage_records = []
        intermediate_maps = {}

        while len(current) > 1 and level < self.max_levels:
            level += 1
            next_level = []
            for index, chunk in enumerate(_chunks(current, self.branching_factor)):
                reduced, usage = await self._reduce_chunk(
                    corpus=corpus,
                    document=document,
                    chunk=chunk,
                    phase=f"reduce-l{level:02d}-c{index + 1:04d}",
                    metadata={
                        "level": level,
                        "chunk_index": index,
                        "chunk_count": len(_chunks(current, self.branching_factor)),
                        "partial_map_count": len(chunk),
                    },
                    services=services,
                    checkpoint_stats=checkpoint_stats,
                )
                reduction_count += 1
                usage_records.append(usage)
                annotated = _annotated_map(
                    reduced,
                    construction_method=self.construction_method,
                    role="reduction",
                    level=level,
                    index=index,
                    extra_facets={"partial_map_count": len(chunk)},
                )
                next_level.append(annotated)
                if len(current) > self.branching_factor:
                    intermediate_maps[
                        _intermediate_map_id(
                            self.id,
                            document.corpus_id,
                            document.id,
                            "reduction",
                            level,
                            index,
                        )
                    ] = annotated
            current = next_level

        if len(current) > 1:
            level += 1
            reduced, usage = await self._reduce_chunk(
                corpus=corpus,
                document=document,
                chunk=current,
                phase=f"reduce-l{level:02d}-final",
                metadata={
                    "level": level,
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "partial_map_count": len(current),
                    "forced_final": True,
                },
                services=services,
                checkpoint_stats=checkpoint_stats,
            )
            reduction_count += 1
            usage_records.append(usage)
            current = [
                _annotated_map(
                    reduced,
                    construction_method=self.construction_method,
                    role="forced_final_reduction",
                    level=level,
                    index=0,
                    extra_facets={"partial_map_count": len(current)},
                )
            ]

        return (
            current[0],
            usage_records,
            {
                "leaf_map_count": len(leaf_maps),
                "reduction_count": reduction_count,
                "levels": level + 1,
            },
            intermediate_maps,
        )

    async def _reduce_chunk(
        self,
        *,
        corpus: Corpus,
        document: Document,
        chunk: list[DocumentMap],
        phase: str,
        metadata: dict[str, int | bool],
        services: Services,
        checkpoint_stats: dict[str, int],
    ) -> tuple[DocumentMap, UsageRecord]:
        signature = input_signature(
            self.id,
            PROMPT_SAFETY_POLICY_VERSION,
            services.map_generation_signature,
            corpus.id,
            document.id,
            self.construction_method,
            phase,
            self.branching_factor,
            self.max_levels,
            [document_map.model_dump_json() for document_map in chunk],
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
            checkpoint_stats["reduction_reused"] += 1
            services.emit_progress(
                f"hierarchical_map reused reduction checkpoint {corpus.id}/{document.id} {phase}"
            )
            return checkpoint.document_map, checkpoint.usage

        try:
            reduced, usage = await reduce_maps_with_model(
                document=document,
                partial_maps=chunk,
                services=services,
                strategy=self.construction_method,
                prompt_parts=("hierarchical_map", "reduce"),
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
                metadata=dict(metadata),
            )
            services.emit_progress(
                "hierarchical_map failed reduction "
                f"{corpus.id}/{document.id} {phase}: {exc.__class__.__name__}: {exc}"
            )
            raise

        write_map_checkpoint(
            services.artifact_store,
            system_id=self.id,
            corpus_id=corpus.id,
            document_id=document.id,
            phase=phase,
            input_signature=signature,
            document_map=reduced,
            usage=usage,
            metadata=dict(metadata),
        )
        checkpoint_stats["reduction_written"] += 1
        services.emit_progress(
            "hierarchical_map wrote reduction checkpoint "
            f"{corpus.id}/{document.id} {phase} "
            f"({usage.model_calls} call(s), "
            f"{usage.input_tokens + usage.output_tokens} token(s))"
        )
        return reduced, usage


def _segment_groups(document: Document, group_size: int) -> list[list[Segment]]:
    segments = sorted(document.segments, key=lambda item: item.order)
    return list(_chunks(segments, group_size))


def _chunks[T](items: list[T], size: int) -> list[list[T]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _annotated_map(
    document_map: DocumentMap,
    *,
    construction_method: str,
    role: str,
    level: int,
    index: int,
    extra_facets: dict[str, int] | None = None,
) -> DocumentMap:
    return document_map.model_copy(
        update={
            "construction_method": construction_method,
            "facets": {
                **document_map.facets,
                "hierarchy_role": role,
                "hierarchy_level": level,
                "hierarchy_index": index,
                **(extra_facets or {}),
            },
        }
    )


def _intermediate_map_id(
    system_id: str,
    corpus_id: str,
    document_id: str,
    role: str,
    level: int,
    index: int,
) -> str:
    return stable_id("map", system_id, corpus_id, document_id, role, level, index)


def _segment_signature_payload(segment: Segment) -> dict[str, object]:
    return {"id": segment.id, "order": segment.order, "text": segment.text}


def _checkpoint_stats() -> dict[str, int]:
    return {
        "leaf_reused": 0,
        "leaf_written": 0,
        "reduction_reused": 0,
        "reduction_written": 0,
    }
