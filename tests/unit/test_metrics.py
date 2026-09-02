from __future__ import annotations

from long_document_indexing.evaluation.local.routing import (
    document_recall_at_k,
    mean_reciprocal_rank,
    required_document_coverage,
)


def test_document_routing_metrics() -> None:
    selected = ["doc_b", "doc_a", "doc_c"]
    relevant = {"doc_a", "doc_c"}

    assert document_recall_at_k(selected, relevant, 1) == 0.0
    assert document_recall_at_k(selected, relevant, 3) == 1.0
    assert mean_reciprocal_rank(selected, relevant) == 0.5
    assert required_document_coverage(selected, relevant) == 1.0
