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
    refine_map_with_model,
)


class RefineSystem(DocumentMapSystemBase):
    """Build a map by processing segments in deterministic order."""

    id = "refine"
    construction_method = "refine"

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
                "refine indexing "
                f"{corpus.id}: document {document_index}/{len(corpus.documents)} "
                f"{document.id} ({len(ordered_segments)} segment(s))"
            )
            current_map, document_usage = await self._refine_document_map(
                corpus=corpus,
                document=document,
                segments=ordered_segments,
                services=services,
                checkpoint_stats=checkpoint_stats,
            )
            usage_records.extend(document_usage)

            if current_map is not None:
                document_maps.append(current_map)
                statuses[document.id] = "indexed"
                services.emit_progress(f"refine indexed {corpus.id}/{document.id}")
            else:
                statuses[document.id] = "empty_document"

        return MapBuildResult(
            document_maps=document_maps,
            usage=combine_usage(*usage_records),
            statuses=statuses,
            metadata={"checkpointing": checkpoint_stats},
        )

    async def _refine_document_map(
        self,
        *,
        corpus: Corpus,
        document: Document,
        segments: list[Segment],
        services: Services,
        checkpoint_stats: dict[str, int],
    ) -> tuple[DocumentMap | None, list[UsageRecord]]:
        current_map: DocumentMap | None = None
        usage_records = []

        for index, segment in enumerate(segments, start=1):
            phase = f"refine-{index:04d}"
            prompt_name = "initial" if current_map is None else "update"
            signature = input_signature(
                self.id,
                PROMPT_SAFETY_POLICY_VERSION,
                corpus.id,
                document.id,
                self.construction_method,
                phase,
                prompt_name,
                segment.id,
                segment.order,
                segment.text,
                current_map.model_dump_json() if current_map is not None else None,
            )
            metadata = {
                "segment_id": segment.id,
                "segment_index": index,
                "segment_count": len(segments),
                "prompt_name": prompt_name,
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
                current_map = checkpoint.document_map
                usage_records.append(checkpoint.usage)
                checkpoint_stats["refinement_reused"] += 1
                services.emit_progress(
                    "refine reused checkpoint "
                    f"{corpus.id}/{document.id} segment {index}/{len(segments)}"
                )
                continue

            try:
                current_map, usage = await refine_map_with_model(
                    document=document,
                    segment=segment,
                    existing_map=current_map,
                    services=services,
                    strategy=self.construction_method,
                    prompt_parts=("refine", prompt_name),
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
                    "refine failed "
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
                document_map=current_map,
                usage=usage,
                metadata=metadata,
            )
            usage_records.append(usage)
            checkpoint_stats["refinement_written"] += 1
            services.emit_progress(
                "refine wrote checkpoint "
                f"{corpus.id}/{document.id} segment {index}/{len(segments)} "
                f"({usage.model_calls} call(s), "
                f"{usage.input_tokens + usage.output_tokens} token(s))"
            )

        return current_map, usage_records


def _checkpoint_stats() -> dict[str, int]:
    return {
        "refinement_reused": 0,
        "refinement_written": 0,
    }
