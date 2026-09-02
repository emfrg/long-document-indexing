from __future__ import annotations

from long_document_indexing.domain.runs import RetrievedItem


def document_recall_at_k(
    selected_document_ids: list[str], relevant_document_ids: set[str], k: int
) -> float:
    if not relevant_document_ids:
        raise ValueError("document recall requires relevant document ids")
    selected = set(selected_document_ids[:k])
    return len(selected & relevant_document_ids) / len(relevant_document_ids)


def mean_reciprocal_rank(
    selected_document_ids: list[str], relevant_document_ids: set[str]
) -> float:
    if not relevant_document_ids:
        raise ValueError("MRR requires relevant document ids")
    for rank, document_id in enumerate(selected_document_ids, start=1):
        if document_id in relevant_document_ids:
            return 1.0 / rank
    return 0.0


def required_document_coverage(
    selected_document_ids: list[str],
    relevant_document_ids: set[str],
) -> float:
    if not relevant_document_ids:
        raise ValueError("required document coverage requires relevant document ids")
    selected = set(selected_document_ids)
    return 1.0 if relevant_document_ids <= selected else 0.0


def segment_recall_at_k(
    retrieved_items: list[RetrievedItem], relevant_segment_ids: set[str], k: int
) -> float:
    if not relevant_segment_ids:
        raise ValueError("segment recall requires relevant segment ids")
    retrieved = {item.segment_id for item in retrieved_items[:k] if item.segment_id is not None}
    return len(retrieved & relevant_segment_ids) / len(relevant_segment_ids)
