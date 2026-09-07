from __future__ import annotations

from long_document_indexing.config import SharedPipelineConfig
from long_document_indexing.domain.corpus import Corpus, Document, Segment
from long_document_indexing.domain.maps import DocumentMap
from long_document_indexing.domain.runs import UsageRecord
from long_document_indexing.services import Services
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

        for document in corpus.documents:
            leaf_maps, leaf_usage = await self._build_leaf_maps(document, services)
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
            ) = await self._reduce_hierarchy(document, leaf_maps, services)
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

        return MapBuildResult(
            document_maps=document_maps,
            usage=combine_usage(*usage_records),
            statuses=statuses,
            intermediate_maps=intermediate_maps,
            metadata={
                "branching_factor": self.branching_factor,
                "max_levels": self.max_levels,
                "documents": documents_metadata,
            },
        )

    async def _build_leaf_maps(
        self,
        document: Document,
        services: Services,
    ) -> tuple[list[DocumentMap], list[UsageRecord]]:
        leaf_maps = []
        usage_records = []
        for index, segments in enumerate(_segment_groups(document, self.branching_factor)):
            document_map, usage = await self.generate_document_map(
                document=document,
                segments=segments,
                services=services,
                prompt_parts=("hierarchical_map", "leaf"),
                extra_metadata={
                    "hierarchy_level": 0,
                    "hierarchy_role": "leaf",
                    "group_index": index,
                    "source_segment_ids": [segment.id for segment in segments],
                },
            )
            leaf_maps.append(document_map)
            usage_records.append(usage)
        return leaf_maps, usage_records

    async def _reduce_hierarchy(
        self,
        document: Document,
        leaf_maps: list[DocumentMap],
        services: Services,
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
                reduced, usage = await reduce_maps_with_model(
                    document=document,
                    partial_maps=chunk,
                    services=services,
                    strategy=self.construction_method,
                    prompt_parts=("hierarchical_map", "reduce"),
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
            reduced, usage = await reduce_maps_with_model(
                document=document,
                partial_maps=current,
                services=services,
                strategy=self.construction_method,
                prompt_parts=("hierarchical_map", "reduce"),
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
