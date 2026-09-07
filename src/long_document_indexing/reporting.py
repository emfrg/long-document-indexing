from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from long_document_indexing.domain.runs import MetricRecord, RagRunRecord
from long_document_indexing.telemetry.usage import UsageEvent

MetricGroup = Literal["routing", "retrieval", "answer", "map", "efficiency", "other"]
UsagePhase = Literal["index", "query", "other"]

ROUTING_METRICS = {
    "document_recall_at_1",
    "document_recall_at_3",
    "mrr",
    "required_document_coverage",
}
RETRIEVAL_METRICS = {"segment_recall_at_4"}
ANSWER_METRICS = {"citation_precision", "citation_recall", "invalid_citation_rate"}
MAP_QUALITY_METRICS = {
    "map_schema_validity",
    "map_source_reference_validity",
    "map_completion_rate",
}
LOWER_IS_BETTER_METRICS = {"invalid_citation_rate", "query_duration_ms", "tool_calls"}
EFFICIENCY_METRICS = {"query_duration_ms", "tool_calls"}


class MetricSummaryRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_id: str
    metric: str
    level: str
    mean: float
    count: int
    group: MetricGroup


class UsageSummaryRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_id: str
    phase: UsagePhase
    kind: str
    records: int
    input_tokens: int
    output_tokens: int
    model_calls: int
    tool_calls: int
    duration_ms: float
    estimated_cost: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class RunStatusSummaryRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_id: str
    total_runs: int
    succeeded: int
    failed: int
    skipped: int


class SystemSummaryRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_id: str
    query_runs: int
    failed_runs: int
    skipped_runs: int
    quality_score: float | None = None
    routing_score: float | None = None
    retrieval_score: float | None = None
    answer_score: float | None = None
    map_score: float | None = None
    query_duration_ms_mean: float | None = None
    tool_calls_mean: float | None = None
    index_model_calls: int = 0
    query_model_calls: int = 0
    total_model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float | None = None
    issues: list[str] = Field(default_factory=list)


class FoundryExportSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    dataset_path: str | None = None
    manifest_path: str | None = None
    row_count: int = 0
    systems: list[str] = Field(default_factory=list)
    evaluators: list[str] = Field(default_factory=list)
    managed_result_path: str | None = None
    managed_row_count: int = 0
    managed_metrics: dict[str, float] = Field(default_factory=dict)
    managed_studio_url: str | None = None


class ReportBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_rows: list[MetricSummaryRow]
    system_rows: list[SystemSummaryRow]
    usage_rows: list[UsageSummaryRow]
    run_status_rows: list[RunStatusSummaryRow]
    foundry_export: FoundryExportSummary


def build_report_bundle(
    *,
    metrics: Iterable[MetricRecord],
    run_records: Iterable[RagRunRecord],
    index_usage: Iterable[UsageEvent],
    query_usage: Iterable[UsageEvent],
    foundry_manifest: Mapping[str, Any] | None = None,
    foundry_manifest_path: str | None = None,
    foundry_managed_result: Mapping[str, Any] | None = None,
    foundry_managed_result_path: str | None = None,
) -> ReportBundle:
    metric_rows = summarize_metrics(metrics)
    usage_rows = summarize_usage_events([*index_usage, *query_usage])
    run_status_rows = summarize_run_status(run_records)
    foundry_export = summarize_foundry_export(
        foundry_manifest,
        foundry_manifest_path,
        foundry_managed_result=foundry_managed_result,
        foundry_managed_result_path=foundry_managed_result_path,
    )
    system_rows = summarize_systems(
        metric_rows=metric_rows,
        usage_rows=usage_rows,
        run_status_rows=run_status_rows,
    )
    return ReportBundle(
        metric_rows=metric_rows,
        system_rows=system_rows,
        usage_rows=usage_rows,
        run_status_rows=run_status_rows,
        foundry_export=foundry_export,
    )


def summarize_metrics(metrics: Iterable[MetricRecord]) -> list[MetricSummaryRow]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for metric in metrics:
        grouped[(metric.system_id, metric.name, metric.level)].append(metric.value)

    rows = []
    for (system_id, name, level), values in sorted(grouped.items()):
        rows.append(
            MetricSummaryRow(
                system_id=system_id,
                metric=name,
                level=level,
                mean=sum(values) / len(values),
                count=len(values),
                group=metric_group(name),
            )
        )
    return rows


def summarize_usage_events(events: Iterable[UsageEvent]) -> list[UsageSummaryRow]:
    grouped: dict[tuple[str, UsagePhase, str], list[UsageEvent]] = defaultdict(list)
    for event in events:
        grouped[(event.system_id, usage_phase(event.stage), event.kind)].append(event)

    rows = []
    for (system_id, phase, kind), group_events in sorted(grouped.items()):
        estimated_costs = [
            event.estimated_cost for event in group_events if event.estimated_cost is not None
        ]
        rows.append(
            UsageSummaryRow(
                system_id=system_id,
                phase=phase,
                kind=kind,
                records=len(group_events),
                input_tokens=sum(event.input_tokens for event in group_events),
                output_tokens=sum(event.output_tokens for event in group_events),
                model_calls=sum(event.model_calls for event in group_events),
                tool_calls=sum(event.tool_calls for event in group_events),
                duration_ms=sum(event.duration_ms for event in group_events),
                estimated_cost=sum(estimated_costs) if estimated_costs else None,
            )
        )
    return rows


def summarize_run_status(records: Iterable[RagRunRecord]) -> list[RunStatusSummaryRow]:
    grouped: dict[str, list[RagRunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.system_id].append(record)

    rows = []
    for system_id, system_records in sorted(grouped.items()):
        rows.append(
            RunStatusSummaryRow(
                system_id=system_id,
                total_runs=len(system_records),
                succeeded=sum(1 for record in system_records if record.status == "succeeded"),
                failed=sum(1 for record in system_records if record.status == "failed"),
                skipped=sum(1 for record in system_records if record.status == "skipped"),
            )
        )
    return rows


def summarize_foundry_export(
    manifest: Mapping[str, Any] | None,
    manifest_path: str | None,
    *,
    foundry_managed_result: Mapping[str, Any] | None = None,
    foundry_managed_result_path: str | None = None,
) -> FoundryExportSummary:
    managed_summary = _managed_result_summary(
        foundry_managed_result,
        foundry_managed_result_path,
    )
    if not manifest:
        return FoundryExportSummary(**managed_summary)

    row_count = manifest.get("row_count", 0)
    systems = manifest.get("systems", [])
    evaluators = manifest.get("evaluators", [])
    return FoundryExportSummary(
        enabled=True,
        dataset_path=_optional_str(manifest.get("dataset_path")),
        manifest_path=manifest_path,
        row_count=int(row_count) if row_count is not None else 0,
        systems=sorted(str(item) for item in systems) if isinstance(systems, list) else [],
        evaluators=[str(item) for item in evaluators] if isinstance(evaluators, list) else [],
        **managed_summary,
    )


def summarize_systems(
    *,
    metric_rows: Iterable[MetricSummaryRow],
    usage_rows: Iterable[UsageSummaryRow],
    run_status_rows: Iterable[RunStatusSummaryRow],
) -> list[SystemSummaryRow]:
    metric_rows = list(metric_rows)
    usage_rows = list(usage_rows)
    run_status_rows = list(run_status_rows)
    system_ids = sorted(
        {
            *(row.system_id for row in metric_rows),
            *(row.system_id for row in usage_rows),
            *(row.system_id for row in run_status_rows),
        }
    )
    status_by_system = {row.system_id: row for row in run_status_rows}

    rows = []
    for system_id in system_ids:
        metrics = [row for row in metric_rows if row.system_id == system_id]
        usage = [row for row in usage_rows if row.system_id == system_id]
        status = status_by_system.get(system_id)
        system_usage = [row for row in usage if row.kind == "system"]
        index_usage = [row for row in system_usage if row.phase == "index"]
        query_usage = [row for row in system_usage if row.phase == "query"]
        input_tokens = sum(row.input_tokens for row in system_usage)
        output_tokens = sum(row.output_tokens for row in system_usage)
        estimated_costs = [
            row.estimated_cost for row in system_usage if row.estimated_cost is not None
        ]
        routing_score = score_metric_group(metrics, "routing")
        retrieval_score = score_metric_group(metrics, "retrieval")
        answer_score = score_metric_group(metrics, "answer")
        map_score = score_metric_group(metrics, "map")
        rows.append(
            SystemSummaryRow(
                system_id=system_id,
                query_runs=status.total_runs if status else 0,
                failed_runs=status.failed if status else 0,
                skipped_runs=status.skipped if status else 0,
                quality_score=quality_score(
                    [routing_score, retrieval_score, answer_score, map_score]
                ),
                routing_score=routing_score,
                retrieval_score=retrieval_score,
                answer_score=answer_score,
                map_score=map_score,
                query_duration_ms_mean=metric_mean(metrics, "query_duration_ms"),
                tool_calls_mean=metric_mean(metrics, "tool_calls"),
                index_model_calls=sum(row.model_calls for row in index_usage),
                query_model_calls=sum(row.model_calls for row in query_usage),
                total_model_calls=sum(row.model_calls for row in system_usage),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                estimated_cost=sum(estimated_costs) if estimated_costs else None,
                issues=system_issues(metrics, status),
            )
        )
    return rows


def metric_group(metric_name: str) -> MetricGroup:
    if metric_name in ROUTING_METRICS:
        return "routing"
    if metric_name in RETRIEVAL_METRICS or metric_name.startswith("segment_recall_at_"):
        return "retrieval"
    if metric_name in ANSWER_METRICS:
        return "answer"
    if metric_name.startswith("map_"):
        return "map"
    if metric_name in EFFICIENCY_METRICS:
        return "efficiency"
    return "other"


def score_metric_group(
    metric_rows: Iterable[MetricSummaryRow],
    group: MetricGroup,
) -> float | None:
    scores = [
        score_metric(row.metric, row.mean)
        for row in metric_rows
        if row.group == group and is_quality_metric(row.metric)
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def score_metric(metric_name: str, value: float) -> float:
    if metric_name == "invalid_citation_rate":
        return max(0.0, 1.0 - value)
    return value


def quality_score(group_scores: Iterable[float | None]) -> float | None:
    scores = [score for score in group_scores if score is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def metric_mean(metric_rows: Iterable[MetricSummaryRow], metric_name: str) -> float | None:
    for row in metric_rows:
        if row.metric == metric_name:
            return row.mean
    return None


def system_issues(
    metric_rows: Iterable[MetricSummaryRow],
    status: RunStatusSummaryRow | None,
) -> list[str]:
    issues = []
    if status is not None:
        if status.failed:
            issues.append(f"{status.failed} failed run(s)")
        if status.skipped:
            issues.append(f"{status.skipped} skipped run(s)")

    for row in metric_rows:
        if row.metric in LOWER_IS_BETTER_METRICS:
            if row.metric == "invalid_citation_rate" and row.mean > 0.0:
                issues.append(f"{row.metric}={row.mean:.4f}")
            continue
        if is_quality_metric(row.metric) and row.mean < 1.0:
            issues.append(f"{row.metric}={row.mean:.4f}")
    return issues


def is_quality_metric(metric_name: str) -> bool:
    if metric_name in EFFICIENCY_METRICS:
        return False
    if metric_name == "map_compression_ratio":
        return False
    return metric_group(metric_name) in {"routing", "retrieval", "answer", "map"}


def usage_phase(stage: str) -> UsagePhase:
    if stage.startswith("index") or stage == "common_indexing":
        return "index"
    if stage.startswith("query") or stage == "common_query":
        return "query"
    return "other"


def legacy_metric_csv_rows(metric_rows: Iterable[MetricSummaryRow]) -> list[dict[str, str | float]]:
    return [
        {
            "system_id": row.system_id,
            "metric": row.metric,
            "mean": row.mean,
            "count": float(row.count),
        }
        for row in metric_rows
    ]


def system_summary_csv_rows(rows: Iterable[SystemSummaryRow]) -> list[dict[str, Any]]:
    return [
        {
            "system_id": row.system_id,
            "query_runs": row.query_runs,
            "failed_runs": row.failed_runs,
            "skipped_runs": row.skipped_runs,
            "quality_score": _csv_float(row.quality_score),
            "routing_score": _csv_float(row.routing_score),
            "retrieval_score": _csv_float(row.retrieval_score),
            "answer_score": _csv_float(row.answer_score),
            "map_score": _csv_float(row.map_score),
            "query_duration_ms_mean": _csv_float(row.query_duration_ms_mean),
            "tool_calls_mean": _csv_float(row.tool_calls_mean),
            "index_model_calls": row.index_model_calls,
            "query_model_calls": row.query_model_calls,
            "total_model_calls": row.total_model_calls,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "total_tokens": row.total_tokens,
            "estimated_cost": _csv_float(row.estimated_cost),
            "issues": "; ".join(row.issues),
        }
        for row in rows
    ]


def usage_summary_csv_rows(rows: Iterable[UsageSummaryRow]) -> list[dict[str, Any]]:
    return [
        {
            "system_id": row.system_id,
            "phase": row.phase,
            "kind": row.kind,
            "records": row.records,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "total_tokens": row.total_tokens,
            "model_calls": row.model_calls,
            "tool_calls": row.tool_calls,
            "duration_ms": row.duration_ms,
            "estimated_cost": _csv_float(row.estimated_cost),
        }
        for row in rows
    ]


def render_markdown_report(bundle: ReportBundle) -> str:
    sections = [
        "# Benchmark Report",
        "",
        "## System Scorecard",
        "",
        _markdown_table(
            [
                "system_id",
                "quality",
                "routing",
                "retrieval",
                "answer",
                "map",
                "query_ms",
                "model_calls",
                "tokens",
                "failed",
            ],
            [
                [
                    row.system_id,
                    _format_float(row.quality_score),
                    _format_float(row.routing_score),
                    _format_float(row.retrieval_score),
                    _format_float(row.answer_score),
                    _format_float(row.map_score),
                    _format_float(row.query_duration_ms_mean, digits=1),
                    str(row.total_model_calls),
                    str(row.total_tokens),
                    str(row.failed_runs),
                ]
                for row in bundle.system_rows
            ],
        ),
        "",
        "## Metric Means",
        "",
        _markdown_table(
            ["system_id", "group", "metric", "mean", "count"],
            [
                [
                    row.system_id,
                    row.group,
                    row.metric,
                    _format_float(row.mean),
                    str(row.count),
                ]
                for row in bundle.metric_rows
            ],
        ),
        "",
        "## Usage Summary",
        "",
        _markdown_table(
            [
                "system_id",
                "phase",
                "kind",
                "records",
                "model_calls",
                "tokens",
                "duration_ms",
            ],
            [
                [
                    row.system_id,
                    row.phase,
                    row.kind,
                    str(row.records),
                    str(row.model_calls),
                    str(row.total_tokens),
                    _format_float(row.duration_ms, digits=1),
                ]
                for row in bundle.usage_rows
            ],
        ),
        "",
        "## Foundry Export",
        "",
        _foundry_export_markdown(bundle.foundry_export),
        "",
        "## Issues",
        "",
        _issues_markdown(bundle.system_rows),
        "",
    ]
    return "\n".join(sections)


def _foundry_export_markdown(summary: FoundryExportSummary) -> str:
    if not summary.enabled:
        if summary.managed_result_path:
            return _foundry_managed_markdown(summary)
        return "No Foundry export manifest was found."
    lines = [
        f"- Dataset: `{summary.dataset_path}`",
        f"- Manifest: `{summary.manifest_path}`",
        f"- Rows: `{summary.row_count}`",
        f"- Systems: `{', '.join(summary.systems)}`",
        f"- Evaluators: `{', '.join(summary.evaluators)}`",
    ]
    if summary.managed_result_path:
        lines.extend(["", _foundry_managed_markdown(summary)])
    return "\n".join(lines)


def _foundry_managed_markdown(summary: FoundryExportSummary) -> str:
    lines = [
        "Managed evaluation:",
        f"- Result: `{summary.managed_result_path}`",
        f"- Rows: `{summary.managed_row_count}`",
    ]
    if summary.managed_studio_url:
        lines.append(f"- Studio URL: `{summary.managed_studio_url}`")
    if summary.managed_metrics:
        lines.append("- Metrics:")
        lines.extend(
            f"  - `{name}`: `{_format_float(value)}`"
            for name, value in sorted(summary.managed_metrics.items())
        )
    return "\n".join(lines)


def _managed_result_summary(
    result: Mapping[str, Any] | None,
    result_path: str | None,
) -> dict[str, Any]:
    if not result:
        return {
            "managed_result_path": None,
            "managed_row_count": 0,
            "managed_metrics": {},
            "managed_studio_url": None,
        }
    row_count = result.get("row_count", 0)
    metrics = result.get("metrics", {})
    return {
        "managed_result_path": result_path,
        "managed_row_count": int(row_count) if row_count is not None else 0,
        "managed_metrics": _float_metrics(metrics),
        "managed_studio_url": _optional_str(result.get("studio_url")),
    }


def _float_metrics(metrics: Any) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        return {}
    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            normalized[str(key)] = float(value)
    return normalized


def _issues_markdown(rows: Iterable[SystemSummaryRow]) -> str:
    lines = []
    for row in rows:
        if row.issues:
            lines.append(f"- `{row.system_id}`: {'; '.join(row.issues)}")
    if not lines:
        return "No local metric issues were detected."
    return "\n".join(lines)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_float(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _csv_float(value: float | None) -> float | str:
    return "" if value is None else value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
