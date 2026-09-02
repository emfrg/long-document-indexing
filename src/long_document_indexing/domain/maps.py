from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SourceReference(BaseModel):
    """Reference from a map entry back to canonical source material."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    segment_ids: list[str] = Field(default_factory=list)


class MapEntry(BaseModel):
    """Normalized retrieval-oriented document-map entry."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    label: str
    summary: str
    source_references: list[SourceReference]
    children: list[MapEntry] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "kind", "label", "summary")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    def walk(self) -> list[MapEntry]:
        entries = [self]
        for child in self.children:
            entries.extend(child.walk())
        return entries


class DocumentMap(BaseModel):
    """Common output shape for every long-document indexing strategy."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    overview: str
    entries: list[MapEntry]
    facets: dict[str, Any] = Field(default_factory=dict)
    construction_method: str

    @field_validator("document_id", "overview", "construction_method")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_entry_ids(self) -> DocumentMap:
        entry_ids = [entry.id for root in self.entries for entry in root.walk()]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError(f"document map for {self.document_id!r} contains duplicate entry ids")
        return self


class IndexArtifact(BaseModel):
    """Persisted index metadata returned by a system's indexing phase."""

    model_config = ConfigDict(extra="forbid")

    id: str
    system_id: str
    corpus_id: str
    artifact_path: str
    document_map_ids: list[str] = Field(default_factory=list)
    build_metadata: dict[str, Any] = Field(default_factory=dict)
