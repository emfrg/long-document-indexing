from __future__ import annotations

from dataclasses import dataclass

from long_document_indexing.retrieval.base import RetrievalBackend
from long_document_indexing.storage.artifacts import ArtifactStore


@dataclass(frozen=True)
class Services:
    """Runtime dependencies passed into systems and runners."""

    artifact_store: ArtifactStore
    retrieval_backend: RetrievalBackend
