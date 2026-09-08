from __future__ import annotations

import math
from typing import Any

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
from long_document_indexing.telemetry.tracing import stable_id


class AgenticMapSystem(DocumentMapSystemBase):
    """Build maps through a bounded inspect-and-revise loop."""

    id = "agentic_map"
    construction_method = "agentic_map"

    def __init__(self, *, max_steps: int = 8, target_coverage: float = 1.0) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not 0.0 < target_coverage <= 1.0:
            raise ValueError("target_coverage must be greater than 0 and at most 1")
        self.max_steps = max_steps
        self.target_coverage = target_coverage

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
                "agentic_map indexing "
                f"{corpus.id}: document {document_index}/{len(corpus.documents)} "
                f"{document.id} ({len(document.segments)} segment(s))"
            )
            final_map, document_usage, action_trace = await self._run_mapping_loop(
                document,
                services,
                corpus_id=corpus.id,
                intermediate_maps=intermediate_maps,
                checkpoint_stats=checkpoint_stats,
            )
            usage_records.extend(document_usage)
            if final_map is None:
                statuses[document.id] = "empty_document"
                documents_metadata[document.id] = {"coverage": 0.0, "steps": 0}
                continue

            coverage = _coverage_ratio(final_map, document.segments)
            statuses[document.id] = (
                "indexed" if coverage >= self.target_coverage else "indexed_partial"
            )
            document_maps.append(
                _annotated_map(
                    final_map,
                    construction_method=self.construction_method,
                    step=len(action_trace),
                    extra_facets={
                        "agent_steps": len(action_trace),
                        "agent_target_coverage": self.target_coverage,
                        "agent_segment_coverage": coverage,
                    },
                )
            )
            documents_metadata[document.id] = {
                "coverage": coverage,
                "steps": len(action_trace),
                "actions": action_trace,
            }
            services.emit_progress(
                "agentic_map indexed "
                f"{corpus.id}/{document.id} coverage={coverage:.3f} "
                f"steps={len(action_trace)}"
            )

        return MapBuildResult(
            document_maps=document_maps,
            usage=combine_usage(*usage_records),
            statuses=statuses,
            intermediate_maps=intermediate_maps,
            metadata={
                "max_steps": self.max_steps,
                "target_coverage": self.target_coverage,
                "documents": documents_metadata,
                "checkpointing": checkpoint_stats,
            },
        )

    async def _run_mapping_loop(
        self,
        document: Document,
        services: Services,
        *,
        corpus_id: str,
        intermediate_maps: dict[str, DocumentMap],
        checkpoint_stats: dict[str, int],
    ) -> tuple[DocumentMap | None, list[UsageRecord], list[dict[str, Any]]]:
        target_segments = _target_segment_count(document.segments, self.target_coverage)
        current_map: DocumentMap | None = None
        usage_records = []
        action_trace = []

        for step in range(1, self.max_steps + 1):
            covered_segment_ids = _covered_segment_ids(current_map)
            if len(covered_segment_ids) >= target_segments:
                break

            segment = _next_uncovered_segment(document.segments, covered_segment_ids)
            if segment is None:
                break

            phase = f"step-{step:04d}"
            prompt_name = "inspect" if current_map is None else "revise"
            metadata = {
                "step": step,
                "selected_segment_id": segment.id,
                "target_segment_count": target_segments,
                "max_steps": self.max_steps,
                "target_coverage": self.target_coverage,
                "prompt_name": prompt_name,
            }
            signature = input_signature(
                self.id,
                PROMPT_SAFETY_POLICY_VERSION,
                corpus_id,
                document.id,
                self.construction_method,
                phase,
                prompt_name,
                self.max_steps,
                self.target_coverage,
                target_segments,
                segment.id,
                segment.order,
                segment.text,
                current_map.model_dump_json() if current_map is not None else None,
            )
            checkpoint = (
                read_map_checkpoint(
                    services.artifact_store,
                    system_id=self.id,
                    corpus_id=corpus_id,
                    document_id=document.id,
                    phase=phase,
                    input_signature=signature,
                )
                if services.resume_checkpoints
                else None
            )
            if checkpoint is not None:
                current_map = checkpoint.document_map
                usage = checkpoint.usage
                checkpoint_stats["step_reused"] += 1
                services.emit_progress(
                    "agentic_map reused step checkpoint "
                    f"{corpus_id}/{document.id} step {step}/{self.max_steps}"
                )
            else:
                try:
                    current_map, usage = await refine_map_with_model(
                        document=document,
                        segment=segment,
                        existing_map=current_map,
                        services=services,
                        strategy=self.construction_method,
                        prompt_parts=("agentic_map", prompt_name),
                    )
                except Exception as exc:
                    write_map_checkpoint_failure(
                        services.artifact_store,
                        system_id=self.id,
                        corpus_id=corpus_id,
                        document_id=document.id,
                        phase=phase,
                        input_signature=signature,
                        error=exc,
                        metadata=metadata,
                    )
                    services.emit_progress(
                        "agentic_map failed "
                        f"{corpus_id}/{document.id} step {step}/{self.max_steps}: "
                        f"{exc.__class__.__name__}: {exc}"
                    )
                    raise

                current_map = _annotated_map(
                    current_map,
                    construction_method=self.construction_method,
                    step=step,
                    extra_facets={
                        "agent_selected_segment_id": segment.id,
                        "agent_target_segments": target_segments,
                    },
                )
                write_map_checkpoint(
                    services.artifact_store,
                    system_id=self.id,
                    corpus_id=corpus_id,
                    document_id=document.id,
                    phase=phase,
                    input_signature=signature,
                    document_map=current_map,
                    usage=usage,
                    metadata=metadata,
                )
                checkpoint_stats["step_written"] += 1
                services.emit_progress(
                    "agentic_map wrote step checkpoint "
                    f"{corpus_id}/{document.id} step {step}/{self.max_steps} "
                    f"({usage.model_calls} call(s), "
                    f"{usage.input_tokens + usage.output_tokens} token(s))"
                )

            usage_records.append(usage)

            covered_after = _covered_segment_ids(current_map)
            action_trace.append(
                {
                    "step": step,
                    "action": "inspect_segment" if step == 1 else "revise_map",
                    "selected_segment_id": segment.id,
                    "covered_segment_count": len(covered_after),
                    "target_segment_count": target_segments,
                }
            )
            intermediate_maps[
                _intermediate_map_id(self.id, corpus_id, document.id, "step", step)
            ] = current_map

        return current_map, usage_records, action_trace


def _target_segment_count(segments: list[Segment], target_coverage: float) -> int:
    return max(1, math.ceil(len(segments) * target_coverage))


def _next_uncovered_segment(
    segments: list[Segment],
    covered_segment_ids: set[str],
) -> Segment | None:
    for segment in sorted(segments, key=lambda item: item.order):
        if segment.id not in covered_segment_ids:
            return segment
    return None


def _covered_segment_ids(document_map: DocumentMap | None) -> set[str]:
    if document_map is None:
        return set()

    segment_ids = set()
    for root in document_map.entries:
        for entry in root.walk():
            for reference in entry.source_references:
                segment_ids.update(reference.segment_ids)
    return segment_ids


def _coverage_ratio(document_map: DocumentMap, segments: list[Segment]) -> float:
    if not segments:
        return 0.0
    segment_ids = {segment.id for segment in segments}
    covered_segment_ids = _covered_segment_ids(document_map)
    return len(segment_ids & covered_segment_ids) / len(segment_ids)


def _annotated_map(
    document_map: DocumentMap,
    *,
    construction_method: str,
    step: int,
    extra_facets: dict[str, object] | None = None,
) -> DocumentMap:
    return document_map.model_copy(
        update={
            "construction_method": construction_method,
            "facets": {
                **document_map.facets,
                "agent_step": step,
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


def _checkpoint_stats() -> dict[str, int]:
    return {
        "step_reused": 0,
        "step_written": 0,
    }
