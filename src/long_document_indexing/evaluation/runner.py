from __future__ import annotations

from collections.abc import Iterable

from long_document_indexing.domain.benchmark import BenchmarkItem
from long_document_indexing.domain.corpus import Corpus
from long_document_indexing.domain.runs import MetricRecord, RagRunRecord
from long_document_indexing.evaluation.local.answers import (
    answer_reference_similarity,
    answer_reference_token_f1,
    answer_reference_token_precision,
    answer_reference_token_recall,
    citation_precision,
    citation_recall,
    invalid_citation_rate,
)
from long_document_indexing.evaluation.local.routing import (
    document_recall_at_k,
    mean_reciprocal_rank,
    required_document_coverage,
    segment_recall_at_k,
)


def evaluate_run(
    record: RagRunRecord,
    item: BenchmarkItem,
    corpus: Corpus,
    metric_names: Iterable[str],
) -> list[MetricRecord]:
    metrics: list[MetricRecord] = []
    for metric_name in metric_names:
        value_and_level = _calculate_metric(metric_name, record, item, corpus)
        if value_and_level is None:
            continue
        value, level = value_and_level
        metrics.append(
            MetricRecord(
                experiment_id=record.experiment_id,
                run_id=record.run_id,
                system_id=record.system_id,
                corpus_id=record.corpus_id,
                item_id=record.item_id,
                repetition=record.repetition,
                level=level,
                name=metric_name,
                value=value,
            )
        )
    return metrics


def aggregate_metric_means(metrics: Iterable[MetricRecord]) -> list[dict[str, str | float]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for metric in metrics:
        grouped.setdefault((metric.system_id, metric.name), []).append(metric.value)

    rows = []
    for (system_id, name), values in sorted(grouped.items()):
        rows.append(
            {
                "system_id": system_id,
                "metric": name,
                "mean": sum(values) / len(values),
                "count": float(len(values)),
            }
        )
    return rows


def _calculate_metric(
    metric_name: str,
    record: RagRunRecord,
    item: BenchmarkItem,
    corpus: Corpus,
) -> tuple[float, str] | None:
    truth = item.ground_truth

    if metric_name.startswith("map_"):
        return None

    if metric_name.startswith("document_recall_at_"):
        if not truth.relevant_document_ids:
            return None
        k = int(metric_name.rsplit("_", maxsplit=1)[1])
        return document_recall_at_k(
            record.selected_document_ids, truth.relevant_document_ids, k
        ), "routing"

    if metric_name == "mrr":
        if not truth.relevant_document_ids:
            return None
        return mean_reciprocal_rank(
            record.selected_document_ids, truth.relevant_document_ids
        ), "routing"

    if metric_name == "required_document_coverage":
        if not truth.relevant_document_ids:
            return None
        return (
            required_document_coverage(record.selected_document_ids, truth.relevant_document_ids),
            "routing",
        )

    if metric_name.startswith("segment_recall_at_"):
        if not truth.relevant_segment_ids:
            return None
        k = int(metric_name.rsplit("_", maxsplit=1)[1])
        return segment_recall_at_k(
            record.retrieved_items, truth.relevant_segment_ids, k
        ), "retrieval"

    if metric_name == "citation_precision":
        if not (truth.relevant_document_ids or truth.relevant_segment_ids):
            return None
        return citation_precision(record.citations, truth), "answer"

    if metric_name == "citation_recall":
        if not (truth.relevant_document_ids or truth.relevant_segment_ids):
            return None
        return citation_recall(record.citations, truth), "answer"

    if metric_name == "invalid_citation_rate":
        return invalid_citation_rate(record.citations, corpus), "answer"

    if metric_name == "answer_reference_similarity":
        if not (truth.reference_summary or truth.expected_answer):
            return None
        return answer_reference_similarity(record.answer, truth), "answer"

    if metric_name == "answer_reference_token_precision":
        if not (truth.reference_summary or truth.expected_answer):
            return None
        return answer_reference_token_precision(record.answer, truth), "answer"

    if metric_name == "answer_reference_token_recall":
        if not (truth.reference_summary or truth.expected_answer):
            return None
        return answer_reference_token_recall(record.answer, truth), "answer"

    if metric_name == "answer_reference_token_f1":
        if not (truth.reference_summary or truth.expected_answer):
            return None
        return answer_reference_token_f1(record.answer, truth), "answer"

    if metric_name == "query_duration_ms":
        return record.usage.duration_ms, "efficiency"

    if metric_name == "tool_calls":
        return float(record.usage.tool_calls), "efficiency"

    raise ValueError(f"unknown local metric: {metric_name}")
