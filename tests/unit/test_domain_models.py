from __future__ import annotations

import pytest
from pydantic import ValidationError

from long_document_indexing.domain.corpus import Corpus, Document, Segment


def test_document_rejects_mismatched_segment_document_id() -> None:
    with pytest.raises(ValidationError):
        Document(
            id="doc_1",
            corpus_id="corpus_1",
            segments=[
                Segment(
                    id="seg_1",
                    document_id="doc_2",
                    order=1,
                    text="Mismatched segment.",
                )
            ],
        )


def test_corpus_indexes_segments_by_id() -> None:
    corpus = Corpus(
        id="corpus_1",
        documents=[
            Document(
                id="doc_1",
                corpus_id="corpus_1",
                segments=[
                    Segment(id="seg_1", document_id="doc_1", order=1, text="First"),
                    Segment(id="seg_2", document_id="doc_1", order=2, text="Second"),
                ],
            )
        ],
    )

    assert set(corpus.segment_by_id()) == {"seg_1", "seg_2"}
