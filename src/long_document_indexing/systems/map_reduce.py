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

REDUCE_FAN_IN = 8


class MapReduceSystem(DocumentMapSystemBase):
    """Build segment-level maps, then reduce them in a bounded fan-in tree."""

    id = "map_reduce"
    construction_method = "map_reduce"

    def __init__(self, *, reduce_fan_in: int = REDUCE_FAN_IN) -> None:
        if reduce_fan_in < 1:
            raise ValueError("reduce_fan_in must be positive")
        self.reduce_fan_in = reduce_fan_in

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
        checkpoint_stats = _checkpoint_stats()

        for document_index, document in enumerate(corpus.documents, start=1):
            ordered_segments = sorted(document.segments, key=lambda item: item.order)
            services.emit_progress(
                "map_reduce indexing "
                f"{corpus.id}: document {document_index}/{len(corpus.documents)} "
                f"{document.id} ({len(ordered_segments)} segment(s))"
            )
            partial_maps, partial_usage = await self._build_partial_maps(
                corpus=corpus,
                document=document,
                segments=ordered_segments,
                services=services,
                checkpoint_stats=checkpoint_stats,
            )
            usage_records.extend(partial_usage)

            if not partial_maps:
                statuses[document.id] = "empty_document"
                continue

            final_map, reduction_usage = await _reduce_tree(
                system_id=self.id,
                corpus=corpus,
                document=document,
                partial_maps=partial_maps,
                services=services,
                strategy=self.construction_method,
                fan_in=self.reduce_fan_in,
                checkpoint_stats=checkpoint_stats,
            )
            document_maps.append(final_map)
            usage_records.append(reduction_usage)
            statuses[document.id] = "indexed"
            services.emit_progress(f"map_reduce indexed {corpus.id}/{document.id}")

        return MapBuildResult(
            document_maps=document_maps,
            usage=combine_usage(*usage_records),
            statuses=statuses,
            metadata={"checkpointing": checkpoint_stats},
        )

    async def _build_partial_maps(
        self,
        *,
        corpus: Corpus,
        document: Document,
        segments: list[Segment],
        services: Services,
        checkpoint_stats: dict[str, int],
    ) -> tuple[list[DocumentMap], list[UsageRecord]]:
        partial_maps = []
        usage_records = []

        for index, segment in enumerate(segments, start=1):
            phase = f"partial-{index:04d}"
            signature = input_signature(
                self.id,
                PROMPT_SAFETY_POLICY_VERSION,
                services.map_generation_signature,
                corpus.id,
                document.id,
                self.construction_method,
                phase,
                segment.id,
                segment.order,
                segment.text,
            )
            metadata = {
                "segment_id": segment.id,
                "segment_index": index,
                "segment_count": len(segments),
            }
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
                partial_maps.append(checkpoint.document_map)
                usage_records.append(checkpoint.usage)
                checkpoint_stats["partial_reused"] += 1
                services.emit_progress(
                    "map_reduce reused checkpoint "
                    f"{corpus.id}/{document.id} segment {index}/{len(segments)}"
                )
                continue

            try:
                partial_map, usage = await self.generate_document_map(
                    document=document,
                    segments=[segment],
                    services=services,
                    prompt_parts=("map_reduce", "map_segment"),
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
                    "map_reduce failed "
                    f"{corpus.id}/{document.id} segment {index}/{len(segments)}: "
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
                document_map=partial_map,
                usage=usage,
                metadata=metadata,
            )
            partial_maps.append(partial_map)
            usage_records.append(usage)
            checkpoint_stats["partial_written"] += 1
            services.emit_progress(
                "map_reduce wrote checkpoint "
                f"{corpus.id}/{document.id} segment {index}/{len(segments)} "
                f"({usage.model_calls} call(s), "
                f"{usage.input_tokens + usage.output_tokens} token(s))"
            )

        return partial_maps, usage_records


async def _reduce_tree(
    *,
    system_id: str,
    corpus: Corpus,
    document: Document,
    partial_maps: list[DocumentMap],
    services: Services,
    strategy: str,
    fan_in: int,
    checkpoint_stats: dict[str, int],
) -> tuple[DocumentMap, UsageRecord]:
    current = partial_maps
    usage_records = []
    level = 0

    while len(current) > 1:
        level += 1
        next_level = []
        chunks = [current[start : start + fan_in] for start in range(0, len(current), fan_in)]
        for chunk_index, chunk in enumerate(chunks, start=1):
            phase = f"reduce-l{level:02d}-c{chunk_index:04d}"
            signature = input_signature(
                system_id,
                PROMPT_SAFETY_POLICY_VERSION,
                services.map_generation_signature,
                corpus.id,
                document.id,
                strategy,
                phase,
                fan_in,
                [document_map.model_dump_json() for document_map in chunk],
            )
            metadata = {
                "level": level,
                "chunk_index": chunk_index,
                "chunk_count": len(chunks),
                "partial_map_count": len(chunk),
            }
            checkpoint = (
                read_map_checkpoint(
                    services.artifact_store,
                    system_id=system_id,
                    corpus_id=corpus.id,
                    document_id=document.id,
                    phase=phase,
                    input_signature=signature,
                )
                if services.resume_checkpoints
                else None
            )
            if checkpoint is not None:
                next_level.append(checkpoint.document_map)
                usage_records.append(checkpoint.usage)
                checkpoint_stats["reduction_reused"] += 1
                services.emit_progress(
                    "map_reduce reused reduction checkpoint "
                    f"{corpus.id}/{document.id} level {level} chunk {chunk_index}/{len(chunks)}"
                )
                continue

            try:
                reduced, usage = await reduce_maps_with_model(
                    document=document,
                    partial_maps=chunk,
                    services=services,
                    strategy=strategy,
                    prompt_parts=("map_reduce", "reduce"),
                )
            except Exception as exc:
                write_map_checkpoint_failure(
                    services.artifact_store,
                    system_id=system_id,
                    corpus_id=corpus.id,
                    document_id=document.id,
                    phase=phase,
                    input_signature=signature,
                    error=exc,
                    metadata=metadata,
                )
                services.emit_progress(
                    "map_reduce failed reduction "
                    f"{corpus.id}/{document.id} level {level} chunk "
                    f"{chunk_index}/{len(chunks)}: {exc.__class__.__name__}: {exc}"
                )
                raise

            write_map_checkpoint(
                services.artifact_store,
                system_id=system_id,
                corpus_id=corpus.id,
                document_id=document.id,
                phase=phase,
                input_signature=signature,
                document_map=reduced,
                usage=usage,
                metadata=metadata,
            )
            next_level.append(reduced)
            usage_records.append(usage)
            checkpoint_stats["reduction_written"] += 1
            services.emit_progress(
                "map_reduce wrote reduction checkpoint "
                f"{corpus.id}/{document.id} level {level} chunk {chunk_index}/{len(chunks)} "
                f"({usage.model_calls} call(s), "
                f"{usage.input_tokens + usage.output_tokens} token(s))"
            )
        current = next_level

    return current[0], combine_usage(*usage_records)


def _checkpoint_stats() -> dict[str, int]:
    return {
        "partial_reused": 0,
        "partial_written": 0,
        "reduction_reused": 0,
        "reduction_written": 0,
    }
