from __future__ import annotations

from long_document_indexing.config import SharedPipelineConfig
from long_document_indexing.domain.corpus import Corpus, Document
from long_document_indexing.domain.maps import DocumentMap
from long_document_indexing.domain.runs import UsageRecord
from long_document_indexing.services import Services
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

        for document in corpus.documents:
            partial_maps = []
            for segment in sorted(document.segments, key=lambda item: item.order):
                partial_map, usage = await self.generate_document_map(
                    document=document,
                    segments=[segment],
                    services=services,
                    prompt_parts=("map_reduce", "map_segment"),
                )
                partial_maps.append(partial_map)
                usage_records.append(usage)

            final_map, reduction_usage = await _reduce_tree(
                document=document,
                partial_maps=partial_maps,
                services=services,
                strategy=self.construction_method,
                fan_in=self.reduce_fan_in,
            )
            document_maps.append(final_map)
            usage_records.append(reduction_usage)
            statuses[document.id] = "indexed"

        return MapBuildResult(
            document_maps=document_maps,
            usage=combine_usage(*usage_records),
            statuses=statuses,
        )


async def _reduce_tree(
    *,
    document: Document,
    partial_maps: list[DocumentMap],
    services: Services,
    strategy: str,
    fan_in: int,
) -> tuple[DocumentMap, UsageRecord]:
    current = partial_maps
    usage_records = []

    while len(current) > 1:
        next_level = []
        for start in range(0, len(current), fan_in):
            chunk = current[start : start + fan_in]
            reduced, usage = await reduce_maps_with_model(
                document=document,
                partial_maps=chunk,
                services=services,
                strategy=strategy,
                prompt_parts=("map_reduce", "reduce"),
            )
            next_level.append(reduced)
            usage_records.append(usage)
        current = next_level

    return current[0], combine_usage(*usage_records)
