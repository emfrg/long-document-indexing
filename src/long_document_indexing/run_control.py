from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from long_document_indexing.config import RunControlConfig
from long_document_indexing.domain.runs import UsageRecord
from long_document_indexing.telemetry.usage import UsageEvent


@dataclass(frozen=True)
class BudgetSnapshot:
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    estimated_cost: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BudgetExceeded(RuntimeError):
    """Raised when a run-control budget is exhausted or exceeded."""


class UnsafeResumeError(RuntimeError):
    """Raised before resume would silently replace stale completed work."""


class BenchmarkIncompleteError(RuntimeError):
    """Raised after persisted query records still contain failures."""


class BudgetLedger:
    """Tracks cumulative usage against configured run-control budgets."""

    def __init__(
        self,
        control: RunControlConfig,
        *,
        initial_events: Iterable[UsageEvent] = (),
    ) -> None:
        self.control = control
        snapshot = snapshot_from_usage_events(initial_events)
        self._input_tokens = snapshot.input_tokens
        self._output_tokens = snapshot.output_tokens
        self._model_calls = snapshot.model_calls
        self._estimated_cost = snapshot.estimated_cost

    @property
    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            model_calls=self._model_calls,
            estimated_cost=self._estimated_cost,
        )

    def add_usage(self, usage: UsageRecord) -> None:
        self._input_tokens += usage.input_tokens
        self._output_tokens += usage.output_tokens
        self._model_calls += usage.model_calls
        if usage.estimated_cost is not None:
            self._estimated_cost = (self._estimated_cost or 0.0) + usage.estimated_cost

    def exhausted_reason(self, label: str) -> str | None:
        snapshot = self.snapshot
        if (
            self.control.max_model_calls is not None
            and snapshot.model_calls >= self.control.max_model_calls
        ):
            return (
                f"Budget exhausted before {label}: model_calls={snapshot.model_calls} "
                f">= max_model_calls={self.control.max_model_calls}"
            )
        if (
            self.control.max_input_tokens is not None
            and snapshot.input_tokens >= self.control.max_input_tokens
        ):
            return (
                f"Budget exhausted before {label}: input_tokens={snapshot.input_tokens} "
                f">= max_input_tokens={self.control.max_input_tokens}"
            )
        if (
            self.control.max_output_tokens is not None
            and snapshot.output_tokens >= self.control.max_output_tokens
        ):
            return (
                f"Budget exhausted before {label}: output_tokens={snapshot.output_tokens} "
                f">= max_output_tokens={self.control.max_output_tokens}"
            )
        if (
            self.control.max_total_tokens is not None
            and snapshot.total_tokens >= self.control.max_total_tokens
        ):
            return (
                f"Budget exhausted before {label}: total_tokens={snapshot.total_tokens} "
                f">= max_total_tokens={self.control.max_total_tokens}"
            )
        if (
            self.control.max_estimated_cost is not None
            and snapshot.estimated_cost is not None
            and snapshot.estimated_cost >= self.control.max_estimated_cost
        ):
            return (
                f"Budget exhausted before {label}: estimated_cost={snapshot.estimated_cost} "
                f">= max_estimated_cost={self.control.max_estimated_cost}"
            )
        return None

    def require_available(self, label: str) -> None:
        reason = self.exhausted_reason(label)
        if reason is not None:
            raise BudgetExceeded(reason)

    def require_estimated_capacity(self, label: str, usage: UsageRecord) -> None:
        """Fail before a phase when observed usage predicts that it cannot finish."""

        projected_input = self._input_tokens + usage.input_tokens
        projected_output = self._output_tokens + usage.output_tokens
        projected_calls = self._model_calls + usage.model_calls
        projected_cost = self._estimated_cost
        if usage.estimated_cost is not None:
            projected_cost = (projected_cost or 0.0) + usage.estimated_cost
        checks = (
            ("model_calls", projected_calls, self.control.max_model_calls),
            ("input_tokens", projected_input, self.control.max_input_tokens),
            ("output_tokens", projected_output, self.control.max_output_tokens),
            (
                "total_tokens",
                projected_input + projected_output,
                self.control.max_total_tokens,
            ),
            ("estimated_cost", projected_cost, self.control.max_estimated_cost),
        )
        for name, projected, limit in checks:
            if limit is not None and projected is not None and projected > limit:
                raise BudgetExceeded(
                    f"Estimated budget insufficient before {label}: projected_{name}="
                    f"{projected} > max_{name}={limit}. No new work was started."
                )

    def require_not_exceeded(self, label: str) -> None:
        snapshot = self.snapshot
        if (
            self.control.max_model_calls is not None
            and snapshot.model_calls > self.control.max_model_calls
        ):
            raise BudgetExceeded(
                f"Budget exceeded after {label}: model_calls={snapshot.model_calls} "
                f"> max_model_calls={self.control.max_model_calls}"
            )
        if (
            self.control.max_input_tokens is not None
            and snapshot.input_tokens > self.control.max_input_tokens
        ):
            raise BudgetExceeded(
                f"Budget exceeded after {label}: input_tokens={snapshot.input_tokens} "
                f"> max_input_tokens={self.control.max_input_tokens}"
            )
        if (
            self.control.max_output_tokens is not None
            and snapshot.output_tokens > self.control.max_output_tokens
        ):
            raise BudgetExceeded(
                f"Budget exceeded after {label}: output_tokens={snapshot.output_tokens} "
                f"> max_output_tokens={self.control.max_output_tokens}"
            )
        if (
            self.control.max_total_tokens is not None
            and snapshot.total_tokens > self.control.max_total_tokens
        ):
            raise BudgetExceeded(
                f"Budget exceeded after {label}: total_tokens={snapshot.total_tokens} "
                f"> max_total_tokens={self.control.max_total_tokens}"
            )
        if (
            self.control.max_estimated_cost is not None
            and snapshot.estimated_cost is not None
            and snapshot.estimated_cost > self.control.max_estimated_cost
        ):
            raise BudgetExceeded(
                f"Budget exceeded after {label}: estimated_cost={snapshot.estimated_cost} "
                f"> max_estimated_cost={self.control.max_estimated_cost}"
            )


def snapshot_from_usage_events(events: Iterable[UsageEvent]) -> BudgetSnapshot:
    events = list(events)
    estimated_costs = [event.estimated_cost for event in events if event.estimated_cost is not None]
    return BudgetSnapshot(
        input_tokens=sum(event.input_tokens for event in events),
        output_tokens=sum(event.output_tokens for event in events),
        model_calls=sum(event.model_calls for event in events),
        estimated_cost=sum(estimated_costs) if estimated_costs else None,
    )


def describe_budget(control: RunControlConfig, snapshot: BudgetSnapshot) -> list[str]:
    lines = [
        f"Current usage: model_calls={snapshot.model_calls}, "
        f"input_tokens={snapshot.input_tokens}, output_tokens={snapshot.output_tokens}, "
        f"total_tokens={snapshot.total_tokens}, estimated_cost={snapshot.estimated_cost}"
    ]
    if not control.has_budget_limits:
        lines.append("Configured budget: none")
        return lines

    lines.append(
        "Configured budget: "
        f"max_model_calls={control.max_model_calls}, "
        f"max_input_tokens={control.max_input_tokens}, "
        f"max_output_tokens={control.max_output_tokens}, "
        f"max_total_tokens={control.max_total_tokens}, "
        f"max_estimated_cost={control.max_estimated_cost}"
    )
    return lines
