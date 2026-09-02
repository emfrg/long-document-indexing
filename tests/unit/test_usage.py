from __future__ import annotations

from long_document_indexing.domain.runs import RunContext, UsageRecord
from long_document_indexing.telemetry.usage import UsageLedger


def test_usage_ledger_records_and_summarizes_usage() -> None:
    context = RunContext(
        experiment_id="exp",
        run_id="run-1",
        system_id="flat_vector",
        corpus_id="corpus",
        item_id="item",
    )
    ledger = UsageLedger()

    ledger.record_usage(
        context,
        stage="query_system",
        kind="system",
        usage=UsageRecord(input_tokens=10, output_tokens=3, model_calls=1, duration_ms=20.0),
    )
    ledger.record_usage(context, stage="common_query", kind="workflow", duration_ms=5.0)

    summary = ledger.summarize("run-1")

    assert summary.input_tokens == 10
    assert summary.output_tokens == 3
    assert summary.model_calls == 1
    assert summary.duration_ms == 25.0
