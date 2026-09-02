from __future__ import annotations

from long_document_indexing.config import SharedPipelineConfig
from long_document_indexing.domain.corpus import Corpus
from long_document_indexing.services import Services
from long_document_indexing.systems.map_base import (
    DocumentMapSystemBase,
    MapBuildResult,
    combine_usage,
    document_source_token_count,
)


class StuffingSystem(DocumentMapSystemBase):
    """Build one map from the full document when it fits the configured budget."""

    id = "stuffing"
    construction_method = "stuffing"

    async def build_document_maps(
        self,
        corpus: Corpus,
        services: Services,
        pipeline: SharedPipelineConfig,
    ) -> MapBuildResult:
        document_maps = []
        usage_records = []
        statuses = {}

        for document in corpus.documents:
            source_tokens = document_source_token_count(document)
            if source_tokens > pipeline.segment_tokens:
                statuses[document.id] = "context_overflow"
                continue

            document_map, usage = await self.generate_document_map(
                document=document,
                segments=document.segments,
                services=services,
                prompt_parts=("stuffing", "map"),
                extra_metadata={"source_tokens": source_tokens},
            )
            document_maps.append(document_map)
            usage_records.append(usage)
            statuses[document.id] = "indexed"

        return MapBuildResult(
            document_maps=document_maps,
            usage=combine_usage(*usage_records),
            statuses=statuses,
        )
