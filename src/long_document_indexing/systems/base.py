from __future__ import annotations

from typing import Protocol

from long_document_indexing.config import SharedPipelineConfig
from long_document_indexing.domain.benchmark import BenchmarkItem
from long_document_indexing.domain.corpus import Corpus
from long_document_indexing.domain.maps import IndexArtifact
from long_document_indexing.domain.runs import RagRunRecord
from long_document_indexing.services import Services


class RagSystem(Protocol):
    id: str

    async def build_index(
        self,
        corpus: Corpus,
        services: Services,
        pipeline: SharedPipelineConfig,
    ) -> IndexArtifact:
        """Build and persist an index artifact for a corpus."""

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
        """Run one benchmark item and return a normalized run record."""
