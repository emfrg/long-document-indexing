from __future__ import annotations

from pathlib import Path
from typing import Protocol

from long_document_indexing.domain.corpus import Corpus
from long_document_indexing.domain.runs import RetrievedItem, UsageRecord


class RetrievalBackend(Protocol):
    async def index(self, corpus: Corpus) -> str:
        """Build or load a retrieval index for a corpus and return its id."""

    async def search(
        self,
        index_id: str,
        query: str,
        *,
        document_ids: list[str] | None,
        top_k: int,
    ) -> list[RetrievedItem]:
        """Search an index, optionally filtering to selected documents."""

    def artifact_path(self, index_id: str) -> Path | None:
        """Return the persisted artifact path when the backend has one."""

    def consume_usage(self) -> UsageRecord:
        """Return and clear model usage accumulated by the last retrieval operation."""
