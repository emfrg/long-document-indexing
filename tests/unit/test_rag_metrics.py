from __future__ import annotations

import pytest

from long_document_indexing.domain.benchmark import EvidenceSpan, GroundTruth
from long_document_indexing.domain.corpus import Corpus, Document, Segment
from long_document_indexing.domain.runs import Citation, RetrievedItem
from long_document_indexing.evaluation.local.rag import (
    citation_support_rate,
    context_precision_at_k,
    context_recall_at_k,
    evidence_quote_recall_at_k,
)


def test_context_metrics_use_segment_labels_when_available() -> None:
    truth = GroundTruth(
        relevant_document_ids={"doc-a"},
        relevant_segment_ids={"seg-a", "seg-b"},
    )
    retrieved = [
        _retrieved("doc-x", "seg-x", rank=1),
        _retrieved("doc-a", "seg-a", rank=2),
        _retrieved("doc-a", "seg-b", rank=3),
    ]

    assert context_precision_at_k(retrieved, truth, 3) == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert context_recall_at_k(retrieved, truth, 3) == 1.0
    assert context_recall_at_k(retrieved, truth, 2) == 0.5


def test_context_metrics_fall_back_to_document_labels() -> None:
    truth = GroundTruth(relevant_document_ids={"doc-a", "doc-b"})
    retrieved = [
        _retrieved("doc-a", "doc-a:s1", rank=1),
        _retrieved("doc-a", "doc-a:s2", rank=2),
        _retrieved("doc-c", "doc-c:s1", rank=3),
    ]

    assert context_precision_at_k(retrieved, truth, 3) == 1.0
    assert context_recall_at_k(retrieved, truth, 3) == 0.5


def test_evidence_quote_recall_checks_retrieved_text() -> None:
    evidence = [
        EvidenceSpan(document_id="doc-a", segment_id="seg-a", quote="ability to pay"),
        EvidenceSpan(document_id="doc-b", segment_id="seg-b", quote="writ of mandamus"),
    ]
    retrieved = [
        _retrieved("doc-a", "seg-a", rank=1, text="The court required an ability to pay inquiry."),
        _retrieved("doc-x", "seg-x", rank=2, text="Unrelated text."),
    ]

    assert evidence_quote_recall_at_k(retrieved, evidence, 2) == 0.5


def test_citation_support_rate_requires_quote_in_cited_source() -> None:
    corpus = Corpus(
        id="case",
        documents=[
            Document(
                id="doc-a",
                corpus_id="case",
                segments=[
                    Segment(
                        id="seg-a",
                        document_id="doc-a",
                        order=1,
                        text="The court discussed a meaningful inquiry into ability to pay.",
                    )
                ],
            )
        ],
    )

    citations = [
        Citation(document_id="doc-a", segment_id="seg-a", quote="meaningful inquiry"),
        Citation(document_id="doc-a", segment_id="seg-a", quote="not in source"),
        Citation(document_id="doc-a", segment_id="missing", quote="meaningful inquiry"),
        Citation(document_id="doc-a"),
    ]

    assert citation_support_rate(citations, corpus) == 0.25


def _retrieved(
    document_id: str,
    segment_id: str,
    *,
    rank: int,
    text: str = "Evidence text.",
) -> RetrievedItem:
    return RetrievedItem(
        document_id=document_id,
        segment_id=segment_id,
        text=text,
        rank=rank,
        retrieval_stage="test",
    )
