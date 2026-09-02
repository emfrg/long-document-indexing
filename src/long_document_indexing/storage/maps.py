from __future__ import annotations

from pathlib import Path

from long_document_indexing.domain.maps import DocumentMap
from long_document_indexing.storage.artifacts import ArtifactStore
from long_document_indexing.telemetry.tracing import stable_id


def document_map_id(system_id: str, corpus_id: str, document_id: str) -> str:
    return stable_id("map", system_id, corpus_id, document_id)


def write_document_map(
    store: ArtifactStore,
    *,
    system_id: str,
    corpus_id: str,
    document_map: DocumentMap,
) -> tuple[str, Path]:
    map_id = document_map_id(system_id, corpus_id, document_map.document_id)
    path = store.write_json(f"indexes/{system_id}/maps/{map_id}.json", document_map)
    return map_id, path


def read_document_map(path: str | Path) -> DocumentMap:
    return DocumentMap.model_validate_json(Path(path).read_text(encoding="utf-8"))
