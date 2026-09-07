from __future__ import annotations

import pytest

from long_document_indexing.config import RunControlConfig
from long_document_indexing.domain.runs import UsageRecord
from long_document_indexing.run_control import (
    BudgetExceeded,
    BudgetLedger,
    describe_budget,
)
from long_document_indexing.telemetry.usage import UsageEvent


def test_budget_ledger_detects_exhausted_model_call_budget() -> None:
    ledger = BudgetLedger(RunControlConfig(max_model_calls=1))

    ledger.require_available("first call")
    ledger.add_usage(UsageRecord(model_calls=1))

    with pytest.raises(BudgetExceeded, match="Budget exhausted before second call"):
        ledger.require_available("second call")


def test_budget_ledger_detects_oversized_single_usage_unit() -> None:
    ledger = BudgetLedger(RunControlConfig(max_total_tokens=10))

    ledger.add_usage(UsageRecord(input_tokens=8, output_tokens=5))

    with pytest.raises(BudgetExceeded, match="Budget exceeded after query"):
        ledger.require_not_exceeded("query")


def test_budget_ledger_can_start_from_existing_usage_events() -> None:
    ledger = BudgetLedger(
        RunControlConfig(max_model_calls=3),
        initial_events=[
            UsageEvent(
                experiment_id="exp",
                run_id="run-1",
                system_id="stuffing",
                corpus_id="corpus",
                stage="index_system",
                kind="system",
                model_calls=2,
                input_tokens=10,
                output_tokens=4,
            )
        ],
    )

    assert ledger.snapshot.model_calls == 2
    assert ledger.snapshot.total_tokens == 14
    assert ledger.exhausted_reason("next call") is None


def test_describe_budget_reports_current_usage_and_limits() -> None:
    lines = describe_budget(
        RunControlConfig(max_model_calls=5),
        BudgetLedger(RunControlConfig(max_model_calls=5)).snapshot,
    )

    assert "Current usage: model_calls=0" in lines[0]
    assert "max_model_calls=5" in lines[1]
