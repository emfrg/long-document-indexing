from __future__ import annotations

import json
from pathlib import Path

from long_document_indexing.domain.maps import DocumentMap
from long_document_indexing.storage.artifacts import ArtifactStore
from long_document_indexing.telemetry.tracing import stable_id

DOCUMENT_MAP_CONTENT_SIGNATURE_POLICY_VERSION = "canonical-json/v1"


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


def document_map_content_signature(document_map: DocumentMap) -> str:
    """Return a stable signature independent of JSON object key ordering."""

    canonical_json = json.dumps(
        document_map.model_dump(mode="json"),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return stable_id("document-map-content", canonical_json)
