from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict

from long_document_indexing.domain.maps import DocumentMap, MapEntry, SourceReference


class StructuredMapAttribute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    value: str


class StructuredSourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    segment_ids: list[str]


class StructuredMapEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    label: str
    summary: str
    source_references: list[StructuredSourceReference]
    children: list[StructuredMapEntry]
    attributes: list[StructuredMapAttribute]

    def to_domain(self) -> MapEntry:
        return MapEntry(
            id=self.id,
            kind=self.kind,
            label=self.label,
            summary=self.summary,
            source_references=[
                SourceReference(
                    document_id=reference.document_id,
                    segment_ids=reference.segment_ids,
                )
                for reference in self.source_references
            ],
            children=[child.to_domain() for child in self.children],
            attributes=_attributes_to_dict(self.attributes),
        )


class StructuredDocumentMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    overview: str
    entries: list[StructuredMapEntry]
    facets: list[StructuredMapAttribute]
    construction_method: str

    def to_document_map(self) -> DocumentMap:
        entries = _deduplicate_entry_ids([entry.to_domain() for entry in self.entries])
        return DocumentMap(
            document_id=self.document_id,
            overview=self.overview,
            entries=entries,
            facets=_attributes_to_dict(self.facets),
            construction_method=self.construction_method,
        )

    def to_generation_content(self) -> str:
        return self.to_document_map().model_dump_json()


def _attributes_to_dict(attributes: list[StructuredMapAttribute]) -> dict[str, str]:
    return {attribute.key: attribute.value for attribute in attributes}


def _deduplicate_entry_ids(entries: list[MapEntry]) -> list[MapEntry]:
    occurrences: defaultdict[str, int] = defaultdict(int)

    def rewrite(entry: MapEntry) -> MapEntry:
        base_id = entry.id.strip() or "entry"
        occurrences[base_id] += 1
        entry_id = base_id
        if occurrences[base_id] > 1:
            entry_id = f"{base_id}_{occurrences[base_id]}"

        return entry.model_copy(
            update={
                "id": entry_id,
                "children": [rewrite(child) for child in entry.children],
            }
        )

    return [rewrite(entry) for entry in entries]


class StructuredAnswerCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    document_id: str
    segment_id: str
    quote: str


class StructuredGeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "insufficient_evidence"]
    answer: str
    citations: list[StructuredAnswerCitation]
