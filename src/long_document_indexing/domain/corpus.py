from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Segment(BaseModel):
    """Smallest canonical source unit available to retrieval and evaluation."""

    model_config = ConfigDict(extra="forbid")

    id: str
    document_id: str
    order: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "document_id", "text")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class Document(BaseModel):
    """A source document containing ordered canonical segments."""

    model_config = ConfigDict(extra="forbid")

    id: str
    corpus_id: str
    title: str | None = None
    segments: list[Segment]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "corpus_id")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_segments(self) -> Document:
        if not self.segments:
            raise ValueError("document must contain at least one segment")

        segment_ids = [segment.id for segment in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError(f"document {self.id!r} contains duplicate segment ids")

        mismatched = [segment.id for segment in self.segments if segment.document_id != self.id]
        if mismatched:
            raise ValueError(
                f"document {self.id!r} contains segments with mismatched document_id: {mismatched}"
            )

        return self


class Corpus(BaseModel):
    """Canonical corpus representation used by all datasets and systems."""

    model_config = ConfigDict(extra="forbid")

    id: str
    documents: list[Document]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_documents(self) -> Corpus:
        if not self.documents:
            raise ValueError("corpus must contain at least one document")

        document_ids = [document.id for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError(f"corpus {self.id!r} contains duplicate document ids")

        mismatched = [document.id for document in self.documents if document.corpus_id != self.id]
        if mismatched:
            raise ValueError(
                f"corpus {self.id!r} contains documents with mismatched corpus_id: {mismatched}"
            )

        return self

    def segment_by_id(self) -> dict[str, Segment]:
        return {
            segment.id: segment
            for document in self.documents
            for segment in sorted(document.segments, key=lambda item: item.order)
        }

    def document_by_id(self) -> dict[str, Document]:
        return {document.id: document for document in self.documents}
