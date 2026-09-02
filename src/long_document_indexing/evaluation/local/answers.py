from __future__ import annotations

from long_document_indexing.domain.benchmark import GroundTruth
from long_document_indexing.domain.corpus import Corpus
from long_document_indexing.domain.runs import Citation


def citation_precision(citations: list[Citation], ground_truth: GroundTruth) -> float:
    if not citations:
        return 0.0

    hits = sum(1 for citation in citations if _citation_hits(citation, ground_truth))
    return hits / len(citations)


def citation_recall(citations: list[Citation], ground_truth: GroundTruth) -> float:
    if ground_truth.relevant_segment_ids:
        cited = {citation.segment_id for citation in citations if citation.segment_id is not None}
        return len(cited & ground_truth.relevant_segment_ids) / len(
            ground_truth.relevant_segment_ids
        )

    if ground_truth.relevant_document_ids:
        cited = {citation.document_id for citation in citations}
        return len(cited & ground_truth.relevant_document_ids) / len(
            ground_truth.relevant_document_ids
        )

    raise ValueError("citation recall requires relevant documents or segments")


def invalid_citation_rate(citations: list[Citation], corpus: Corpus) -> float:
    if not citations:
        return 0.0

    document_ids = set(corpus.document_by_id())
    segment_ids = set(corpus.segment_by_id())

    invalid_count = 0
    for citation in citations:
        if citation.document_id not in document_ids:
            invalid_count += 1
            continue
        if citation.segment_id is not None and citation.segment_id not in segment_ids:
            invalid_count += 1

    return invalid_count / len(citations)


def _citation_hits(citation: Citation, ground_truth: GroundTruth) -> bool:
    if ground_truth.relevant_segment_ids:
        return citation.segment_id in ground_truth.relevant_segment_ids
    if ground_truth.relevant_document_ids:
        return citation.document_id in ground_truth.relevant_document_ids
    return False
