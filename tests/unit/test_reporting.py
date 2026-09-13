from __future__ import annotations

import pytest

from long_document_indexing.domain.runs import (
    Citation,
    MetricRecord,
    RagRunRecord,
    RetrievedItem,
)
from long_document_indexing.reporting import (
    build_report_bundle,
    confidence_interval_csv_rows,
    legacy_metric_csv_rows,
    metric_group,
    render_markdown_report,
    system_summary_csv_rows,
)
from long_document_indexing.telemetry.usage import UsageEvent


def test_report_bundle_summarizes_quality_usage_and_foundry_export() -> None:
    bundle = build_report_bundle(
        metrics=[
            _metric("alpha", "document_recall_at_1", 1.0, level="routing"),
            _metric("alpha", "citation_precision", 1.0, level="answer"),
            _metric("alpha", "invalid_citation_rate", 0.0, level="answer"),
            _metric("alpha", "query_duration_ms", 20.0, level="efficiency"),
            _metric("beta", "document_recall_at_1", 0.5, level="routing"),
            _metric("beta", "citation_precision", 0.5, level="answer"),
            _metric("beta", "invalid_citation_rate", 0.25, level="answer"),
            _metric("beta", "map_schema_validity", 0.75, level="document_map"),
            _metric("beta", "map_compression_ratio", 3.0, level="document_map"),
        ],
        run_records=[
            _run_record("alpha", status="succeeded"),
            _run_record("beta", status="succeeded"),
            _run_record("beta", status="failed"),
        ],
        index_usage=[
            _usage("alpha", stage="index_system", model_calls=1, input_tokens=100),
            _usage("beta", stage="index_system", model_calls=2, input_tokens=200),
        ],
        query_usage=[
            _usage("alpha", stage="query_system", model_calls=1, output_tokens=30),
            _usage("beta", stage="query_system", model_calls=1, output_tokens=40),
        ],
        foundry_manifest={
            "dataset_path": "evaluations/foundry/dataset.jsonl",
            "row_count": 3,
            "systems": ["beta", "alpha"],
            "evaluators": ["groundedness"],
        },
        foundry_manifest_path="/tmp/manifest.json",
        foundry_managed_result={
            "evaluation_name": "benchmark-evaluation",
            "azure_ai_project": (
                "/subscriptions/sub/resourceGroups/rg/providers/"
                "Microsoft.CognitiveServices/accounts/account/projects/project-name"
            ),
            "row_count": 3,
            "metrics": {"f1.f1_score": 0.8, "rouge.rouge": 0.7},
            "studio_url": "https://ai.azure.com/resource/build/evaluation/legacy-run",
        },
        foundry_managed_result_path="/tmp/managed-result.json",
    )

    alpha = _system(bundle.system_rows, "alpha")
    beta = _system(bundle.system_rows, "beta")

    assert alpha.quality_score == 1.0
    assert alpha.total_model_calls == 2
    assert alpha.total_tokens == 130
    assert beta.routing_score == 0.5
    assert beta.answer_score == pytest.approx(0.625)
    assert beta.map_score == 0.75
    assert beta.quality_score == pytest.approx(0.625)
    assert beta.failed_runs == 1
    assert "1 failed run(s)" in beta.issues
    assert "invalid_citation_rate=0.2500" in beta.issues
    assert bundle.foundry_export.enabled is True
    assert bundle.foundry_export.row_count == 3
    assert bundle.foundry_export.systems == ["alpha", "beta"]
    assert bundle.foundry_export.managed_result_path == "/tmp/managed-result.json"
    assert bundle.foundry_export.managed_row_count == 3
    assert bundle.foundry_export.managed_metrics == {
        "f1.f1_score": 0.8,
        "rouge.rouge": 0.7,
    }

    markdown = render_markdown_report(bundle)
    assert "Managed evaluation:" in markdown
    assert "`f1.f1_score`: `0.8000`" in markdown
    assert "Foundry project: `project-name`" in markdown
    assert "Evaluation: `benchmark-evaluation`" in markdown
    assert "[Microsoft Foundry](https://ai.azure.com/)" in markdown
    assert "/resource/build/evaluation/" not in markdown


def test_report_renderers_keep_legacy_metric_csv_and_add_scorecard_rows() -> None:
    bundle = build_report_bundle(
        metrics=[_metric("alpha", "document_recall_at_1", 1.0, level="routing")],
        run_records=[_run_record("alpha", status="succeeded")],
        index_usage=[],
        query_usage=[],
    )

    assert legacy_metric_csv_rows(bundle.metric_rows) == [
        {
            "system_id": "alpha",
            "metric": "document_recall_at_1",
            "mean": 1.0,
            "count": 1.0,
        }
    ]
    assert system_summary_csv_rows(bundle.system_rows)[0]["quality_score"] == 1.0
    assert confidence_interval_csv_rows(bundle.metric_rows)[0] == {
        "system_id": "alpha",
        "group": "routing",
        "metric": "document_recall_at_1",
        "mean": 1.0,
        "ci95_low": 1.0,
        "ci95_high": 1.0,
        "count": 1.0,
    }
    markdown = render_markdown_report(bundle)
    assert "## System Scorecard" in markdown
    assert "## Metric Means" in markdown
    assert "No Foundry export manifest was found." in markdown


def test_report_computes_deterministic_bootstrap_confidence_intervals() -> None:
    metrics = [
        MetricRecord(
            experiment_id="exp",
            run_id=f"run-{index}",
            system_id="alpha",
            corpus_id=f"corpus-{index}",
            item_id=f"item-{index}",
            level="retrieval",
            name="context_recall_at_4",
            value=value,
        )
        for index, value in enumerate([0.0, 0.0, 1.0, 1.0])
    ]

    first = build_report_bundle(
        metrics=metrics,
        run_records=[],
        index_usage=[],
        query_usage=[],
    ).metric_rows[0]
    second = build_report_bundle(
        metrics=metrics,
        run_records=[],
        index_usage=[],
        query_usage=[],
    ).metric_rows[0]

    assert first.mean == 0.5
    assert first.ci95_low == 0.0
    assert first.ci95_high == 1.0
    assert first == second
    assert "[0.0000, 1.0000]" in render_markdown_report(
        build_report_bundle(
            metrics=metrics,
            run_records=[],
            index_usage=[],
            query_usage=[],
        )
    )


def test_report_renders_per_system_managed_metrics() -> None:
    bundle = build_report_bundle(
        metrics=[],
        run_records=[],
        index_usage=[],
        query_usage=[],
        foundry_managed_result={
            "evaluation_name": "managed",
            "row_count": 4,
            "metrics": {"groundedness.score": 3.0},
            "metadata": {
                "scope_results": {
                    "flat_vector": {
                        "metrics": {"groundedness.score": 2.0},
                    },
                    "map_reduce": {
                        "metrics": {"groundedness.score": 4.0},
                    },
                }
            },
        },
        foundry_managed_result_path="/tmp/managed-result.json",
    )

    assert bundle.foundry_export.managed_system_metrics == {
        "flat_vector": {"groundedness.score": 2.0},
        "map_reduce": {"groundedness.score": 4.0},
    }
    markdown = render_markdown_report(bundle)
    assert "Per-system managed metrics" in markdown
    assert "| flat_vector | 2.0000 |" in markdown


def test_report_marks_managed_evaluation_unavailable_for_current_export() -> None:
    bundle = build_report_bundle(
        metrics=[],
        run_records=[],
        index_usage=[],
        query_usage=[],
        foundry_manifest={
            "dataset_path": "evaluations/foundry/dataset.jsonl",
            "row_count": 0,
            "systems": [],
            "evaluators": ["groundedness"],
        },
    )

    assert (
        "Managed evaluation: not available for the current export."
        in render_markdown_report(bundle)
    )


def test_report_scores_answer_reference_metrics_without_requiring_perfection() -> None:
    bundle = build_report_bundle(
        metrics=[
            _metric("alpha", "answer_reference_token_f1", 0.45, level="answer"),
            _metric("alpha", "invalid_citation_rate", 0.0, level="answer"),
        ],
        run_records=[_run_record("alpha", status="succeeded")],
        index_usage=[],
        query_usage=[],
    )

    row = _system(bundle.system_rows, "alpha")

    assert metric_group("answer_reference_token_f1") == "answer"
    assert row.answer_score == pytest.approx(0.725)
    assert row.quality_score == pytest.approx(0.725)
    assert "answer_reference_token_f1=0.4500" not in row.issues


def test_report_groups_rag_specific_metrics() -> None:
    bundle = build_report_bundle(
        metrics=[
            _metric("alpha", "context_precision_at_4", 0.75, level="retrieval"),
            _metric("alpha", "context_recall_at_4", 0.5, level="retrieval"),
            _metric("alpha", "evidence_quote_recall_at_4", 0.25, level="retrieval"),
            _metric("alpha", "citation_support_rate", 1.0, level="answer"),
        ],
        run_records=[_run_record("alpha", status="succeeded")],
        index_usage=[],
        query_usage=[],
    )

    row = _system(bundle.system_rows, "alpha")

    assert metric_group("context_precision_at_4") == "retrieval"
    assert metric_group("evidence_quote_recall_at_4") == "retrieval"
    assert metric_group("citation_support_rate") == "answer"
    assert row.retrieval_score == pytest.approx(0.5)
    assert row.answer_score == 1.0
    assert "evidence_quote_recall_at_4=0.2500" in row.issues


def _metric(system_id: str, name: str, value: float, *, level: str) -> MetricRecord:
    return MetricRecord(
        experiment_id="exp",
        run_id=f"run-{system_id}-{name}",
        system_id=system_id,
        corpus_id="corpus",
        item_id="item",
        level=level,
        name=name,
        value=value,
    )


def _usage(
    system_id: str,
    *,
    stage: str,
    model_calls: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> UsageEvent:
    return UsageEvent(
        experiment_id="exp",
        run_id=f"run-{system_id}-{stage}",
        system_id=system_id,
        corpus_id="corpus",
        stage=stage,
        kind="system",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_calls=model_calls,
    )


def _run_record(system_id: str, *, status: str) -> RagRunRecord:
    return RagRunRecord(
        run_id=f"run-{system_id}-{status}",
        experiment_id="exp",
        system_id=system_id,
        corpus_id="corpus",
        item_id="item",
        selected_document_ids=["doc"],
        retrieved_items=[
            RetrievedItem(
                document_id="doc",
                segment_id="seg",
                text="Evidence.",
                rank=1,
                retrieval_stage="test",
            )
        ],
        answer="Answer.",
        citations=[Citation(document_id="doc", segment_id="seg")],
        status=status,
    )


def _system(rows, system_id: str):
    return next(row for row in rows if row.system_id == system_id)
