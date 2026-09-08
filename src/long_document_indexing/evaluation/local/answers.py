from __future__ import annotations

from collections import Counter

from long_document_indexing.domain.benchmark import GroundTruth
from long_document_indexing.domain.corpus import Corpus
from long_document_indexing.domain.runs import Citation
from long_document_indexing.text import lexical_similarity, tokenize


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


def answer_reference_similarity(answer: str, ground_truth: GroundTruth) -> float:
    reference = _reference_text(ground_truth)
    return lexical_similarity(answer, reference)


def answer_reference_token_precision(answer: str, ground_truth: GroundTruth) -> float:
    answer_token_count, _reference_token_count, overlap = _reference_token_overlap(
        answer,
        ground_truth,
    )
    if answer_token_count == 0:
        return 0.0
    return overlap / answer_token_count


def answer_reference_token_recall(answer: str, ground_truth: GroundTruth) -> float:
    _answer_token_count, reference_token_count, overlap = _reference_token_overlap(
        answer,
        ground_truth,
    )
    if reference_token_count == 0:
        return 0.0
    return overlap / reference_token_count


def answer_reference_token_f1(answer: str, ground_truth: GroundTruth) -> float:
    precision = answer_reference_token_precision(answer, ground_truth)
    recall = answer_reference_token_recall(answer, ground_truth)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _citation_hits(citation: Citation, ground_truth: GroundTruth) -> bool:
    if ground_truth.relevant_segment_ids:
        return citation.segment_id in ground_truth.relevant_segment_ids
    if ground_truth.relevant_document_ids:
        return citation.document_id in ground_truth.relevant_document_ids
    return False


def _reference_token_overlap(answer: str, ground_truth: GroundTruth) -> tuple[int, int, int]:
    answer_counts = Counter(tokenize(answer))
    reference_counts = Counter(tokenize(_reference_text(ground_truth)))
    overlap = sum(
        min(answer_counts[token], reference_counts[token])
        for token in answer_counts.keys() & reference_counts.keys()
    )
    return sum(answer_counts.values()), sum(reference_counts.values()), overlap


def _reference_text(ground_truth: GroundTruth) -> str:
    reference = ground_truth.reference_summary or ground_truth.expected_answer
    if reference is None or not reference.strip():
        raise ValueError("answer reference metrics require expected_answer or reference_summary")
    return reference
