from __future__ import annotations

from long_document_indexing.config import SharedPipelineConfig
from long_document_indexing.domain.corpus import Corpus
from long_document_indexing.domain.maps import DocumentMap
from long_document_indexing.services import Services
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

        for document in corpus.documents:
            current_map: DocumentMap | None = None
            for segment in sorted(document.segments, key=lambda item: item.order):
                current_map, usage = await refine_map_with_model(
                    document=document,
                    segment=segment,
                    existing_map=current_map,
                    services=services,
                    strategy=self.construction_method,
                    prompt_parts=("refine", "initial" if current_map is None else "update"),
                )
                usage_records.append(usage)

            if current_map is not None:
                document_maps.append(current_map)
                statuses[document.id] = "indexed"
            else:
                statuses[document.id] = "empty_document"

        return MapBuildResult(
            document_maps=document_maps,
            usage=combine_usage(*usage_records),
            statuses=statuses,
        )
