from __future__ import annotations

import pytest

from long_document_indexing.domain.benchmark import BenchmarkItem, GroundTruth
from long_document_indexing.domain.corpus import Corpus, Document, Segment
from long_document_indexing.domain.runs import RagRunRecord, UsageRecord
from long_document_indexing.evaluation.local.answers import (
    answer_reference_similarity,
    answer_reference_token_f1,
    answer_reference_token_precision,
    answer_reference_token_recall,
)
from long_document_indexing.evaluation.local.routing import (
    document_recall_at_k,
    mean_reciprocal_rank,
    required_document_coverage,
)
from long_document_indexing.evaluation.runner import evaluate_run


def test_document_routing_metrics() -> None:
    selected = ["doc_b", "doc_a", "doc_c"]
    relevant = {"doc_a", "doc_c"}

    assert document_recall_at_k(selected, relevant, 1) == 0.0
    assert document_recall_at_k(selected, relevant, 3) == 1.0
    assert mean_reciprocal_rank(selected, relevant) == 0.5
    assert required_document_coverage(selected, relevant) == 1.0


def test_answer_reference_metrics_use_reference_summary() -> None:
    truth = GroundTruth(reference_summary="Alpha beta beta gamma.")
    answer = "Alpha beta delta."

    assert answer_reference_token_precision(answer, truth) == pytest.approx(2 / 3)
    assert answer_reference_token_recall(answer, truth) == pytest.approx(2 / 4)
    assert answer_reference_token_f1(answer, truth) == pytest.approx(4 / 7)
    assert answer_reference_similarity(answer, truth) > 0.0


def test_evaluate_run_emits_answer_reference_metrics() -> None:
    item = BenchmarkItem(
        id="q1",
        corpus_id="corpus",
        query="Summarize the case.",
        ground_truth=GroundTruth(reference_summary="Alpha beta beta gamma."),
    )
    corpus = Corpus(
        id="corpus",
        documents=[
            Document(
                id="doc",
                corpus_id="corpus",
                segments=[
                    Segment(
                        id="seg",
                        document_id="doc",
                        order=1,
                        text="Alpha beta delta.",
                    )
                ],
            )
        ],
    )
    record = RagRunRecord(
        run_id="run",
        experiment_id="exp",
        system_id="system",
        corpus_id="corpus",
        item_id="q1",
        selected_document_ids=["doc"],
        retrieved_items=[],
        answer="Alpha beta delta.",
        citations=[],
        usage=UsageRecord(duration_ms=3.0),
        status="succeeded",
    )

    metrics = evaluate_run(
        record,
        item,
        corpus,
        [
            "answer_reference_similarity",
            "answer_reference_token_precision",
            "answer_reference_token_recall",
            "answer_reference_token_f1",
        ],
    )

    assert [metric.name for metric in metrics] == [
        "answer_reference_similarity",
        "answer_reference_token_precision",
        "answer_reference_token_recall",
        "answer_reference_token_f1",
    ]
    assert {metric.level for metric in metrics} == {"answer"}
