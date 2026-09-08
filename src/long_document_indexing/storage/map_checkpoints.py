from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from long_document_indexing.domain.maps import DocumentMap
from long_document_indexing.domain.runs import UsageRecord
from long_document_indexing.storage.artifacts import ArtifactStore
from long_document_indexing.telemetry.tracing import stable_id


class MapCheckpoint(BaseModel):
    """Reusable checkpoint for one completed map-generation step."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str
    system_id: str
    corpus_id: str
    document_id: str
    phase: str
    input_signature: str
    document_map: DocumentMap
    usage: UsageRecord = Field(default_factory=UsageRecord)
    metadata: dict[str, Any] = Field(default_factory=dict)


def input_signature(*parts: object) -> str:
    """Stable fingerprint for checkpoint inputs."""

    return stable_id("map-input", *parts, length=32)


def map_checkpoint_id(
    *,
    system_id: str,
    corpus_id: str,
    document_id: str,
    phase: str,
    input_signature: str,
) -> str:
    return stable_id(
        "map-checkpoint",
        system_id,
        corpus_id,
        document_id,
        phase,
        input_signature,
    )


def read_map_checkpoint(
    store: ArtifactStore,
    *,
    system_id: str,
    corpus_id: str,
    document_id: str,
    phase: str,
    input_signature: str,
) -> MapCheckpoint | None:
    checkpoint_id = map_checkpoint_id(
        system_id=system_id,
        corpus_id=corpus_id,
        document_id=document_id,
        phase=phase,
        input_signature=input_signature,
    )
    path = store.experiment_dir / _checkpoint_relative_path(
        system_id=system_id,
        corpus_id=corpus_id,
        document_id=document_id,
        checkpoint_id=checkpoint_id,
    )
    if not path.exists():
        return None

    checkpoint = MapCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
    if checkpoint.input_signature != input_signature:
        return None
    return checkpoint


def write_map_checkpoint(
    store: ArtifactStore,
    *,
    system_id: str,
    corpus_id: str,
    document_id: str,
    phase: str,
    input_signature: str,
    document_map: DocumentMap,
    usage: UsageRecord,
    metadata: dict[str, Any] | None = None,
) -> MapCheckpoint:
    checkpoint = MapCheckpoint(
        checkpoint_id=map_checkpoint_id(
            system_id=system_id,
            corpus_id=corpus_id,
            document_id=document_id,
            phase=phase,
            input_signature=input_signature,
        ),
        system_id=system_id,
        corpus_id=corpus_id,
        document_id=document_id,
        phase=phase,
        input_signature=input_signature,
        document_map=document_map,
        usage=usage,
        metadata=metadata or {},
    )
    store.write_json(
        _checkpoint_relative_path(
            system_id=system_id,
            corpus_id=corpus_id,
            document_id=document_id,
            checkpoint_id=checkpoint.checkpoint_id,
        ),
        checkpoint,
    )
    return checkpoint


def write_map_checkpoint_failure(
    store: ArtifactStore,
    *,
    system_id: str,
    corpus_id: str,
    document_id: str,
    phase: str,
    input_signature: str,
    error: Exception,
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint_id = map_checkpoint_id(
        system_id=system_id,
        corpus_id=corpus_id,
        document_id=document_id,
        phase=phase,
        input_signature=input_signature,
    )
    store.write_json(
        _checkpoint_relative_path(
            system_id=system_id,
            corpus_id=corpus_id,
            document_id=document_id,
            checkpoint_id=checkpoint_id,
            failure=True,
        ),
        {
            "checkpoint_id": checkpoint_id,
            "system_id": system_id,
            "corpus_id": corpus_id,
            "document_id": document_id,
            "phase": phase,
            "input_signature": input_signature,
            "error_type": error.__class__.__name__,
            "error": str(error),
            "metadata": metadata or {},
        },
    )


def _checkpoint_relative_path(
    *,
    system_id: str,
    corpus_id: str,
    document_id: str,
    checkpoint_id: str,
    failure: bool = False,
) -> str:
    document_bucket = stable_id("document", corpus_id, document_id)
    leaf_dir = "failures" if failure else "completed"
    return f"indexes/{system_id}/checkpoints/{document_bucket}/{leaf_dir}/{checkpoint_id}.json"
