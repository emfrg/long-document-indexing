from __future__ import annotations

import re

from long_document_indexing.domain.benchmark import EvidenceSpan, GroundTruth
from long_document_indexing.domain.corpus import Corpus
from long_document_indexing.domain.runs import Citation, RetrievedItem


def context_precision_at_k(
    retrieved_items: list[RetrievedItem],
    ground_truth: GroundTruth,
    k: int,
) -> float:
    """Rank-aware precision over relevant retrieved contexts.

    This mirrors the average-precision style used by RAGAS context precision:
    irrelevant material above relevant material lowers the score, while missing
    relevant material is handled separately by context recall.
    """

    relevant_ids = _relevant_context_ids(ground_truth)
    if not relevant_ids:
        raise ValueError("context precision requires relevant documents or segments")

    hits = 0
    precision_sum = 0.0
    seen_ids: set[str] = set()
    for rank, item in enumerate(retrieved_items[:k], start=1):
        context_id = _retrieved_context_id(item, ground_truth)
        if context_id is None or context_id in seen_ids:
            continue
        seen_ids.add(context_id)
        if context_id not in relevant_ids:
            continue
        hits += 1
        precision_sum += hits / rank

    if hits == 0:
        return 0.0
    return precision_sum / hits


def context_recall_at_k(
    retrieved_items: list[RetrievedItem],
    ground_truth: GroundTruth,
    k: int,
) -> float:
    relevant_ids = _relevant_context_ids(ground_truth)
    if not relevant_ids:
        raise ValueError("context recall requires relevant documents or segments")

    retrieved_ids = {
        _retrieved_context_id(item, ground_truth)
        for item in retrieved_items[:k]
        if _retrieved_context_id(item, ground_truth) is not None
    }
    return len(retrieved_ids & relevant_ids) / len(relevant_ids)


def evidence_quote_recall_at_k(
    retrieved_items: list[RetrievedItem],
    evidence: list[EvidenceSpan],
    k: int,
) -> float:
    quoted_evidence = [span for span in evidence if span.quote and span.quote.strip()]
    if not quoted_evidence:
        raise ValueError("evidence quote recall requires evidence quotes")

    retrieved_texts = [
        _normalized_text(item.text)
        for item in retrieved_items[:k]
        if item.text and item.text.strip()
    ]
    hits = sum(
        1
        for span in quoted_evidence
        if any(_normalized_text(span.quote or "") in text for text in retrieved_texts)
    )
    return hits / len(quoted_evidence)


def citation_support_rate(citations: list[Citation], corpus: Corpus) -> float:
    """Fraction of citations whose quote is present in the cited source location."""

    if not citations:
        return 0.0

    segments = corpus.segment_by_id()
    documents = corpus.document_by_id()
    supported = 0
    for citation in citations:
        if not citation.quote or not citation.quote.strip():
            continue

        source_text = ""
        if citation.segment_id is not None:
            segment = segments.get(citation.segment_id)
            if segment is None or segment.document_id != citation.document_id:
                continue
            source_text = segment.text
        else:
            document = documents.get(citation.document_id)
            if document is None:
                continue
            source_text = "\n".join(segment.text for segment in document.segments)

        if _normalized_text(citation.quote) in _normalized_text(source_text):
            supported += 1

    return supported / len(citations)


def _relevant_context_ids(ground_truth: GroundTruth) -> set[str]:
    if ground_truth.relevant_segment_ids:
        return set(ground_truth.relevant_segment_ids)
    return set(ground_truth.relevant_document_ids)


def _retrieved_context_id(
    item: RetrievedItem,
    ground_truth: GroundTruth,
) -> str | None:
    if ground_truth.relevant_segment_ids:
        return item.segment_id
    return item.document_id


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()
