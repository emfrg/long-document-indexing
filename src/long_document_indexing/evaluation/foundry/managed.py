from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

from long_document_indexing.config import ExperimentConfig, FoundryEvaluationConfig, ModelConfig
from long_document_indexing.evaluation.foundry.adapter import dataset_sha256
from long_document_indexing.storage.artifacts import ArtifactStore

_TEXT_SCORE_EVALUATORS = {"f1", "rouge", "bleu", "gleu", "meteor"}
_MODEL_JUDGE_EVALUATORS = {
    "groundedness",
    "relevance",
    "retrieval",
    "response_completeness",
    "qa",
    "similarity",
}
_DOCUMENT_RETRIEVAL_EVALUATORS = {"document_retrieval"}
_SUPPORTED_MANAGED_EVALUATORS = (
    sorted(_TEXT_SCORE_EVALUATORS | _MODEL_JUDGE_EVALUATORS | _DOCUMENT_RETRIEVAL_EVALUATORS)
)
_REASONING_MODEL_DEPLOYMENT_PATTERN = re.compile(
    r"(?:^|[^a-z0-9])(?:gpt[-_]?5|o[134])(?:$|[^a-z0-9])",
    re.IGNORECASE,
)
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
_FALSEY_ENV_VALUES = {"0", "false", "no", "n", "off"}
_MANAGED_EVALUATOR_CHECKPOINT_DIR = Path("evaluations/foundry/managed-evaluators")
_TRANSIENT_ERROR_MARKERS = (
    "429",
    "apiconnectionerror",
    "apitimeouterror",
    "connection error",
    "connection reset",
    "rate limit",
    "readerror",
    "readtimeout",
    "service request error",
    "service response error",
    "temporarily unavailable",
    "timeout",
)


class ManagedEvaluationIncompleteError(RuntimeError):
    """Raised when an SDK result cannot prove complete evaluator coverage."""


class FoundryManagedEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_name: str
    dataset_path: str
    result_path: str
    azure_ai_project: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    row_count: int = 0
    studio_url: str | None = None
    oai_eval_run_ids: list[dict[str, str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def run_foundry_managed_evaluation(
    *,
    config: ExperimentConfig,
    store: ArtifactStore,
    evaluate_fn: Callable[..., Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    status_callback: Callable[[str], None] | None = None,
) -> FoundryManagedEvaluationResult:
    """Run Azure AI Evaluation SDK over an exported Foundry JSONL dataset."""

    foundry_config = config.evaluation.foundry
    dataset_path = store.experiment_dir / foundry_config.dataset_path
    manifest_path = store.experiment_dir / foundry_config.manifest_path
    result_path = store.experiment_dir / foundry_config.result_path

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Foundry evaluation dataset not found: {dataset_path}. "
            "Run `ldi evaluate` or `ldi export-foundry-eval` first."
        )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Foundry evaluation manifest not found: {manifest_path}. "
            "Run `ldi evaluate` or `ldi export-foundry-eval` first."
        )

    manifest = store.read_json(foundry_config.manifest_path)
    actual_dataset_sha256 = dataset_sha256(dataset_path)
    expected_dataset_sha256 = manifest.get("dataset_sha256")
    if expected_dataset_sha256 != actual_dataset_sha256:
        raise ValueError(
            "Foundry evaluation dataset does not match its manifest. "
            "Run `ldi evaluate` or `ldi export-foundry-eval` again."
        )
    expected_row_count = _dataset_row_count(dataset_path)
    manifest_row_count = manifest.get("row_count")
    if manifest_row_count != expected_row_count:
        raise ValueError(
            "Foundry evaluation row count does not match its manifest. "
            "Run `ldi evaluate` or `ldi export-foundry-eval` again."
        )

    azure_ai_project = _azure_ai_project(foundry_config)
    evaluation_name = foundry_config.evaluation_name or config.experiment.id
    evaluate = evaluate_fn or _load_evaluate()

    if (
        foundry_config.managed_group_by != "none"
        or foundry_config.managed_sample_fraction < 1.0
    ):
        normalized, evaluated_row_count = _run_managed_evaluation_scopes(
            config=config,
            source_dataset_path=dataset_path,
            result_path=result_path,
            azure_ai_project=azure_ai_project,
            evaluation_name=evaluation_name,
            evaluate=evaluate,
            use_placeholder_evaluators=evaluate_fn is not None,
            store=store,
            sleep_fn=sleep_fn,
            status_callback=status_callback,
        )
    elif foundry_config.managed_execution == "sequential":
        normalized = _run_managed_evaluators_sequentially(
            config=config,
            dataset_path=dataset_path,
            result_path=result_path,
            azure_ai_project=azure_ai_project,
            evaluation_name=evaluation_name,
            evaluate=evaluate,
            use_placeholder_evaluators=evaluate_fn is not None,
            store=store,
            dataset_sha256_value=actual_dataset_sha256,
            expected_row_count=expected_row_count,
            sleep_fn=sleep_fn,
            status_callback=status_callback,
        )
        evaluated_row_count = expected_row_count
    else:
        normalized = _run_managed_evaluators_in_parallel(
            config=config,
            dataset_path=dataset_path,
            result_path=result_path,
            azure_ai_project=azure_ai_project,
            evaluation_name=evaluation_name,
            evaluate=evaluate,
            use_placeholder_evaluators=evaluate_fn is not None,
            expected_row_count=expected_row_count,
            sleep_fn=sleep_fn,
            status_callback=status_callback,
        )
        evaluated_row_count = expected_row_count
    normalized.metadata["dataset_sha256"] = actual_dataset_sha256
    normalized.metadata["manifest_schema_version"] = manifest.get("schema_version")
    if any(
        name.lower() in _MODEL_JUDGE_EVALUATORS
        for name in foundry_config.managed_evaluators
    ):
        normalized.metadata["judge_deployment"] = _resolved_string(
            config.models.judge_deployment
        )
    normalized.metadata["completion_status"] = "complete"
    normalized.metadata["source_row_count"] = expected_row_count
    normalized.metadata["expected_row_count"] = evaluated_row_count
    _validate_complete_managed_result(
        normalized,
        evaluator_names=foundry_config.managed_evaluators,
        expected_row_count=evaluated_row_count,
    )
    store.write_json(foundry_config.result_path, normalized)
    return normalized


def _run_managed_evaluators_in_parallel(
    *,
    config: ExperimentConfig,
    dataset_path: Path,
    result_path: Path,
    azure_ai_project: str | None,
    evaluation_name: str,
    evaluate: Callable[..., Any],
    use_placeholder_evaluators: bool,
    expected_row_count: int,
    sleep_fn: Callable[[float], None],
    status_callback: Callable[[str], None] | None,
    scope_tags: Mapping[str, str] | None = None,
) -> FoundryManagedEvaluationResult:
    foundry_config = config.evaluation.foundry
    normalized, attempts = _evaluate_with_retries(
        config=config,
        evaluate=evaluate,
        dataset_path=dataset_path,
        result_path=result_path,
        evaluator_names=foundry_config.managed_evaluators,
        evaluation_name=evaluation_name,
        azure_ai_project=azure_ai_project,
        tags={**_evaluation_tags(config, foundry_config), **dict(scope_tags or {})},
        use_placeholder_evaluators=use_placeholder_evaluators,
        expected_row_count=expected_row_count,
        sleep_fn=sleep_fn,
        status_callback=status_callback,
    )
    normalized.metadata["managed_execution"] = "parallel"
    normalized.metadata["attempts"] = attempts
    return normalized


def _run_managed_evaluators_sequentially(
    *,
    config: ExperimentConfig,
    dataset_path: Path,
    result_path: Path,
    azure_ai_project: str | None,
    evaluation_name: str,
    evaluate: Callable[..., Any],
    use_placeholder_evaluators: bool,
    store: ArtifactStore,
    dataset_sha256_value: str,
    expected_row_count: int,
    sleep_fn: Callable[[float], None],
    status_callback: Callable[[str], None] | None,
    checkpoint_group: str | None = None,
    scope_tags: Mapping[str, str] | None = None,
) -> FoundryManagedEvaluationResult:
    foundry_config = config.evaluation.foundry
    metrics: dict[str, Any] = {}
    row_count = 0
    studio_url: str | None = None
    oai_eval_run_ids: list[dict[str, str]] = []
    evaluator_results: list[dict[str, Any]] = []

    for index, evaluator_name in enumerate(foundry_config.managed_evaluators):
        evaluator_evaluation_name = (
            evaluation_name
            if len(foundry_config.managed_evaluators) == 1
            else f"{evaluation_name}-{_evaluation_name_slug(evaluator_name)}"
        )
        tags = {**_evaluation_tags(config, foundry_config), **dict(scope_tags or {})}
        tags["managed_evaluator"] = evaluator_name
        checkpoint_signature = _managed_evaluator_signature(
            config=config,
            evaluator_name=evaluator_name,
            dataset_sha256_value=dataset_sha256_value,
        )
        checkpoint_path = _managed_evaluator_checkpoint_path(
            evaluator_name,
            group=checkpoint_group,
        )
        normalized = _load_managed_evaluator_checkpoint(
            store=store,
            checkpoint_path=checkpoint_path,
            checkpoint_signature=checkpoint_signature,
            evaluator_name=evaluator_name,
            expected_row_count=expected_row_count,
        )
        reused = normalized is not None
        attempts = 0
        if normalized is None:
            _emit_status(
                status_callback,
                f"Running evaluator {index + 1}/{len(foundry_config.managed_evaluators)}: "
                f"{evaluator_name}",
            )
            normalized, attempts = _evaluate_with_retries(
                config=config,
                evaluate=evaluate,
                dataset_path=dataset_path,
                result_path=result_path,
                evaluator_names=[evaluator_name],
                evaluation_name=evaluator_evaluation_name,
                azure_ai_project=azure_ai_project,
                tags=tags,
                use_placeholder_evaluators=use_placeholder_evaluators,
                expected_row_count=expected_row_count,
                sleep_fn=sleep_fn,
                status_callback=status_callback,
            )
            normalized.metadata.update(
                {
                    "checkpoint_signature": checkpoint_signature,
                    "completion_status": "complete",
                    "dataset_sha256": dataset_sha256_value,
                    "evaluator": evaluator_name,
                    "expected_row_count": expected_row_count,
                }
            )
            store.write_json(checkpoint_path, normalized)
            _emit_status(status_callback, f"Checkpointed evaluator: {evaluator_name}")
        else:
            _emit_status(status_callback, f"Reusing completed evaluator: {evaluator_name}")
        metrics.update(normalized.metrics)
        row_count = max(row_count, normalized.row_count)
        studio_url = studio_url or normalized.studio_url
        oai_eval_run_ids.extend(normalized.oai_eval_run_ids)
        evaluator_results.append(
            {
                "evaluator": evaluator_name,
                "evaluation_name": evaluator_evaluation_name,
                "row_count": normalized.row_count,
                "studio_url": normalized.studio_url,
                "oai_eval_run_ids": normalized.oai_eval_run_ids,
                "metrics": normalized.metrics,
                "completion_status": "complete",
                "checkpoint_path": str(store.experiment_dir / checkpoint_path),
                "reused": reused,
                "attempts": attempts,
            }
        )
        should_delay = index < len(foundry_config.managed_evaluators) - 1
        if (
            not reused
            and should_delay
            and foundry_config.managed_evaluator_delay_seconds > 0
        ):
            _emit_status(
                status_callback,
                f"Waiting {foundry_config.managed_evaluator_delay_seconds:g}s before "
                "the next evaluator",
            )
            sleep_fn(foundry_config.managed_evaluator_delay_seconds)

    return FoundryManagedEvaluationResult(
        evaluation_name=evaluation_name,
        dataset_path=str(dataset_path),
        result_path=str(result_path),
        azure_ai_project=azure_ai_project,
        metrics=metrics,
        row_count=row_count,
        studio_url=studio_url,
        oai_eval_run_ids=oai_eval_run_ids,
        metadata={
            "sdk_result_type": "sequential",
            "managed_execution": "sequential",
            "managed_evaluator_delay_seconds": foundry_config.managed_evaluator_delay_seconds,
            "evaluator_results": evaluator_results,
        },
    )


def _run_managed_evaluation_scopes(
    *,
    config: ExperimentConfig,
    source_dataset_path: Path,
    result_path: Path,
    azure_ai_project: str | None,
    evaluation_name: str,
    evaluate: Callable[..., Any],
    use_placeholder_evaluators: bool,
    store: ArtifactStore,
    sleep_fn: Callable[[float], None],
    status_callback: Callable[[str], None] | None,
) -> tuple[FoundryManagedEvaluationResult, int]:
    foundry_config = config.evaluation.foundry
    source_rows = _read_managed_dataset_rows(source_dataset_path)
    sampled_rows, selected_item_ids = _sample_managed_rows(
        source_rows,
        fraction=foundry_config.managed_sample_fraction,
        seed=foundry_config.managed_sample_seed,
    )
    rows_by_scope = _managed_rows_by_scope(
        sampled_rows,
        group_by=foundry_config.managed_group_by,
    )
    if foundry_config.managed_group_by == "system_id":
        _validate_scope_item_coverage(rows_by_scope, selected_item_ids)
    system_results: dict[str, Any] = {}
    scoped_results: list[FoundryManagedEvaluationResult] = []

    for scope_index, (scope_name, rows) in enumerate(sorted(rows_by_scope.items()), start=1):
        scope_slug = _evaluation_name_slug(scope_name)
        scope_dataset_path = store.write_jsonl(
            Path("evaluations/foundry/managed-samples") / f"{scope_slug}.jsonl",
            rows,
        )
        scope_evaluation_name = (
            evaluation_name
            if scope_name == "all"
            else f"{evaluation_name}-{scope_slug}"
        )
        scope_tags = {
            "managed_group_by": foundry_config.managed_group_by,
            "managed_scope": scope_name,
        }
        _emit_status(
            status_callback,
            f"Running managed scope {scope_index}/{len(rows_by_scope)}: "
            f"{scope_name} ({len(rows)} rows)",
        )
        if foundry_config.managed_execution == "sequential":
            scoped = _run_managed_evaluators_sequentially(
                config=config,
                dataset_path=scope_dataset_path,
                result_path=result_path,
                azure_ai_project=azure_ai_project,
                evaluation_name=scope_evaluation_name,
                evaluate=evaluate,
                use_placeholder_evaluators=use_placeholder_evaluators,
                store=store,
                dataset_sha256_value=dataset_sha256(scope_dataset_path),
                expected_row_count=len(rows),
                sleep_fn=sleep_fn,
                status_callback=status_callback,
                checkpoint_group=scope_name,
                scope_tags=scope_tags,
            )
        else:
            scoped = _run_managed_evaluators_in_parallel(
                config=config,
                dataset_path=scope_dataset_path,
                result_path=result_path,
                azure_ai_project=azure_ai_project,
                evaluation_name=scope_evaluation_name,
                evaluate=evaluate,
                use_placeholder_evaluators=use_placeholder_evaluators,
                expected_row_count=len(rows),
                sleep_fn=sleep_fn,
                status_callback=status_callback,
                scope_tags=scope_tags,
            )
        scoped_results.append(scoped)
        system_results[scope_name] = {
            "dataset_path": str(scope_dataset_path),
            "evaluation_name": scoped.evaluation_name,
            "metrics": scoped.metrics,
            "row_count": scoped.row_count,
            "studio_url": scoped.studio_url,
            "oai_eval_run_ids": scoped.oai_eval_run_ids,
            "metadata": scoped.metadata,
        }

    evaluated_row_count = sum(result.row_count for result in scoped_results)
    return (
        FoundryManagedEvaluationResult(
            evaluation_name=evaluation_name,
            dataset_path=str(source_dataset_path),
            result_path=str(result_path),
            azure_ai_project=azure_ai_project,
            metrics=_mean_managed_metrics(scoped_results),
            row_count=evaluated_row_count,
            studio_url=next(
                (result.studio_url for result in scoped_results if result.studio_url),
                None,
            ),
            oai_eval_run_ids=[
                run_id
                for result in scoped_results
                for run_id in result.oai_eval_run_ids
            ],
            metadata={
                "sdk_result_type": "scoped",
                "managed_execution": foundry_config.managed_execution,
                "managed_group_by": foundry_config.managed_group_by,
                "managed_sample_fraction": foundry_config.managed_sample_fraction,
                "managed_sample_seed": foundry_config.managed_sample_seed,
                "selected_item_count": len(selected_item_ids),
                "selected_item_ids": selected_item_ids,
                "scope_results": system_results,
            },
        ),
        evaluated_row_count,
    )


def _read_managed_dataset_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as dataset:
        for line_number, line in enumerate(dataset, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"managed dataset row {line_number} must be an object")
            rows.append(row)
    return rows


def _sample_managed_rows(
    rows: list[dict[str, Any]],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    items: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("item_id") or row.get("id") or "").strip()
        if not item_id:
            raise ValueError("managed dataset rows must include item_id or id")
        items.setdefault(item_id, row)

    if fraction >= 1.0:
        selected_item_ids = sorted(items)
    else:
        strata: dict[str, list[str]] = defaultdict(list)
        for item_id, row in items.items():
            strata[_managed_row_stratum(row)].append(item_id)
        selected_item_ids = []
        for stratum, item_ids in sorted(strata.items()):
            ranked = sorted(
                item_ids,
                key=lambda item_id: _stable_sample_key(seed, stratum, item_id),
            )
            sample_size = max(1, math.ceil(len(ranked) * fraction))
            selected_item_ids.extend(ranked[:sample_size])
        selected_item_ids.sort()

    selected = set(selected_item_ids)
    return (
        [row for row in rows if str(row.get("item_id") or row.get("id")) in selected],
        selected_item_ids,
    )


def _managed_row_stratum(row: Mapping[str, Any]) -> str:
    tags = {str(tag) for tag in row.get("tags", [])}
    if "single_hop" in tags:
        return "single_hop"
    if "chained_multi_hop" in tags:
        return "chained_multi_hop"
    if "multi_hop" in tags:
        return "multi_hop"
    return "other"


def _stable_sample_key(seed: int, stratum: str, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{stratum}:{item_id}".encode()).hexdigest()


def _managed_rows_by_scope(
    rows: list[dict[str, Any]],
    *,
    group_by: str,
) -> dict[str, list[dict[str, Any]]]:
    if group_by == "none":
        return {"all": rows}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        system_id = str(row.get("system_id") or "").strip()
        if not system_id:
            raise ValueError("managed_group_by=system_id requires system_id on every row")
        grouped[system_id].append(row)
    return dict(grouped)


def _validate_scope_item_coverage(
    rows_by_scope: Mapping[str, list[dict[str, Any]]],
    selected_item_ids: list[str],
) -> None:
    expected = set(selected_item_ids)
    for scope_name, rows in rows_by_scope.items():
        actual = {str(row.get("item_id") or row.get("id")) for row in rows}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"managed scope {scope_name!r} does not cover the shared item sample; "
                f"missing={missing}, extra={extra}"
            )


def _mean_managed_metrics(
    results: list[FoundryManagedEvaluationResult],
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for name, value in result.metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[name].append(float(value))
    return {
        name: sum(metric_values) / len(metric_values)
        for name, metric_values in sorted(values.items())
        if metric_values
    }


def _evaluate_with_retries(
    *,
    config: ExperimentConfig,
    evaluate: Callable[..., Any],
    dataset_path: Path,
    result_path: Path,
    evaluator_names: list[str],
    evaluation_name: str,
    azure_ai_project: str | None,
    tags: dict[str, str],
    use_placeholder_evaluators: bool,
    expected_row_count: int,
    sleep_fn: Callable[[float], None],
    status_callback: Callable[[str], None] | None,
) -> tuple[FoundryManagedEvaluationResult, int]:
    foundry_config = config.evaluation.foundry
    names = ", ".join(evaluator_names)
    for attempt in range(1, foundry_config.managed_max_attempts + 1):
        _emit_status(
            status_callback,
            f"Attempt {attempt}/{foundry_config.managed_max_attempts}: {names}",
        )
        try:
            result = _call_evaluate(
                evaluate=evaluate,
                dataset_path=dataset_path,
                evaluator_names=evaluator_names,
                evaluation_name=evaluation_name,
                azure_ai_project=azure_ai_project,
                fail_on_evaluator_errors=foundry_config.fail_on_evaluator_errors,
                tags=tags,
                config=config,
                use_placeholder_evaluators=use_placeholder_evaluators,
            )
            normalized = _normalize_result(
                result,
                evaluation_name=evaluation_name,
                dataset_path=dataset_path,
                result_path=result_path,
                azure_ai_project=azure_ai_project,
            )
            _validate_complete_managed_result(
                normalized,
                evaluator_names=evaluator_names,
                expected_row_count=expected_row_count,
            )
            _emit_status(status_callback, f"Completed evaluator(s): {names}")
            return normalized, attempt
        except Exception as exc:
            can_retry = attempt < foundry_config.managed_max_attempts and (
                isinstance(exc, ManagedEvaluationIncompleteError)
                or _is_transient_managed_error(exc)
            )
            if not can_retry:
                raise RuntimeError(
                    f"managed evaluator(s) {names} did not complete after {attempt} "
                    f"attempt(s): {exc}"
                ) from exc
            delay = foundry_config.managed_retry_delay_seconds * (2 ** (attempt - 1))
            _emit_status(
                status_callback,
                f"Retrying evaluator(s) {names} after {delay:g}s "
                f"(attempt {attempt + 1}/{foundry_config.managed_max_attempts})",
            )
            sleep_fn(delay)
    raise AssertionError("managed evaluation retry loop terminated unexpectedly")


def _validate_complete_managed_result(
    result: FoundryManagedEvaluationResult,
    *,
    evaluator_names: list[str],
    expected_row_count: int,
) -> None:
    if result.row_count != expected_row_count:
        raise ManagedEvaluationIncompleteError(
            f"expected {expected_row_count} rows, received {result.row_count}"
        )
    missing_metrics = [
        evaluator_name
        for evaluator_name in evaluator_names
        if not _has_evaluator_metric(result.metrics, evaluator_name)
    ]
    if missing_metrics:
        raise ManagedEvaluationIncompleteError(
            "missing aggregate metrics for evaluator(s): " + ", ".join(missing_metrics)
        )


def _has_evaluator_metric(metrics: Mapping[str, Any], evaluator_name: str) -> bool:
    prefix = f"{evaluator_name.lower()}."
    return any(
        str(name).lower() == evaluator_name.lower()
        or str(name).lower().startswith(prefix)
        for name in metrics
    )


def _is_transient_managed_error(exc: Exception) -> bool:
    message = f"{exc.__class__.__name__}: {exc}".lower()
    return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)


def _emit_status(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _dataset_row_count(path: Path) -> int:
    with path.open(encoding="utf-8") as dataset:
        return sum(1 for line in dataset if line.strip())


def _managed_evaluator_checkpoint_path(
    evaluator_name: str,
    *,
    group: str | None = None,
) -> Path:
    filename = f"{_evaluation_name_slug(evaluator_name)}.json"
    if group is None:
        return _MANAGED_EVALUATOR_CHECKPOINT_DIR / filename
    return _MANAGED_EVALUATOR_CHECKPOINT_DIR / _evaluation_name_slug(group) / filename


def _managed_evaluator_signature(
    *,
    config: ExperimentConfig,
    evaluator_name: str,
    dataset_sha256_value: str,
) -> str:
    payload = {
        "dataset_sha256": dataset_sha256_value,
        "evaluator": evaluator_name,
        "evaluator_config": _managed_evaluator_config([evaluator_name]),
        "judge_deployment": (
            _resolved_string(config.models.judge_deployment)
            if evaluator_name.lower() in _MODEL_JUDGE_EVALUATORS
            else None
        ),
        "reasoning_model": (
            _is_reasoning_model_judge(config.models)
            if evaluator_name.lower() in _MODEL_JUDGE_EVALUATORS
            else None
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"managed-evaluator-{digest[:16]}"


def _load_managed_evaluator_checkpoint(
    *,
    store: ArtifactStore,
    checkpoint_path: Path,
    checkpoint_signature: str,
    evaluator_name: str,
    expected_row_count: int,
) -> FoundryManagedEvaluationResult | None:
    path = store.experiment_dir / checkpoint_path
    if not path.exists():
        return None
    try:
        result = FoundryManagedEvaluationResult.model_validate(store.read_json(checkpoint_path))
    except (OSError, ValueError):
        return None
    if result.metadata.get("checkpoint_signature") != checkpoint_signature:
        return None
    if result.metadata.get("completion_status") != "complete":
        return None
    try:
        _validate_complete_managed_result(
            result,
            evaluator_names=[evaluator_name],
            expected_row_count=expected_row_count,
        )
    except ManagedEvaluationIncompleteError:
        return None
    return result


def _call_evaluate(
    *,
    evaluate: Callable[..., Any],
    dataset_path: Path,
    evaluator_names: list[str],
    evaluation_name: str,
    azure_ai_project: str | None,
    fail_on_evaluator_errors: bool,
    tags: dict[str, str],
    config: ExperimentConfig,
    use_placeholder_evaluators: bool,
) -> Any:
    evaluators = (
        _placeholder_evaluators(evaluator_names)
        if use_placeholder_evaluators
        else _managed_evaluators(evaluator_names, config=config)
    )
    return evaluate(
        data=str(dataset_path),
        evaluators=evaluators,
        evaluation_name=evaluation_name,
        evaluator_config=_managed_evaluator_config(evaluator_names),
        azure_ai_project=azure_ai_project,
        fail_on_evaluator_errors=fail_on_evaluator_errors,
        tags=tags,
    )


def build_foundry_managed_evaluation_plan(
    *,
    config: ExperimentConfig,
    store: ArtifactStore,
) -> dict[str, Any]:
    """Return the SDK call shape without importing Azure packages or executing evaluation."""

    foundry_config = config.evaluation.foundry
    dataset_path = store.experiment_dir / foundry_config.dataset_path
    manifest_path = store.experiment_dir / foundry_config.manifest_path
    azure_ai_project = _azure_ai_project(foundry_config)
    sample_plan = _managed_sample_plan(dataset_path, foundry_config)
    return {
        "evaluation_name": foundry_config.evaluation_name or config.experiment.id,
        "dataset_path": str(dataset_path),
        "dataset_exists": dataset_path.exists(),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "result_path": str(store.experiment_dir / foundry_config.result_path),
        "azure_ai_project": azure_ai_project,
        "managed_evaluators": foundry_config.managed_evaluators,
        "managed_execution": foundry_config.managed_execution,
        "managed_group_by": foundry_config.managed_group_by,
        "managed_sample_fraction": foundry_config.managed_sample_fraction,
        "managed_sample_seed": foundry_config.managed_sample_seed,
        **sample_plan,
        "managed_evaluator_delay_seconds": foundry_config.managed_evaluator_delay_seconds,
        "managed_max_attempts": foundry_config.managed_max_attempts,
        "managed_retry_delay_seconds": foundry_config.managed_retry_delay_seconds,
        "judge_model_config": _judge_model_config_plan(
            config.models,
            evaluator_names=foundry_config.managed_evaluators,
        ),
        "evaluator_config": _managed_evaluator_config(foundry_config.managed_evaluators),
        "fail_on_evaluator_errors": foundry_config.fail_on_evaluator_errors,
        "tags": _evaluation_tags(config, foundry_config),
    }


def _managed_sample_plan(
    dataset_path: Path,
    config: FoundryEvaluationConfig,
) -> dict[str, Any]:
    if not dataset_path.exists():
        return {
            "source_row_count": None,
            "selected_item_count": None,
            "evaluated_row_count": None,
            "scope_row_counts": {},
        }
    rows = _read_managed_dataset_rows(dataset_path)
    sampled_rows, selected_item_ids = _sample_managed_rows(
        rows,
        fraction=config.managed_sample_fraction,
        seed=config.managed_sample_seed,
    )
    rows_by_scope = _managed_rows_by_scope(sampled_rows, group_by=config.managed_group_by)
    return {
        "source_row_count": len(rows),
        "selected_item_count": len(selected_item_ids),
        "evaluated_row_count": len(sampled_rows),
        "scope_row_counts": {
            scope: len(scope_rows) for scope, scope_rows in sorted(rows_by_scope.items())
        },
    }


def _load_evaluate() -> Callable[..., Any]:
    try:
        from azure.ai.evaluation import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "Azure AI Evaluation SDK is not installed. "
            "Install the optional Foundry dependencies with "
            "`uv sync --python 3.13 --extra dev --extra foundry`."
        ) from exc
    return evaluate


def _managed_evaluators(
    names: list[str],
    *,
    config: ExperimentConfig,
) -> dict[str, Callable[..., Any]]:
    try:
        from azure.ai.evaluation import (
            BleuScoreEvaluator,
            DocumentRetrievalEvaluator,
            F1ScoreEvaluator,
            GleuScoreEvaluator,
            GroundednessEvaluator,
            MeteorScoreEvaluator,
            QAEvaluator,
            RelevanceEvaluator,
            ResponseCompletenessEvaluator,
            RetrievalEvaluator,
            RougeScoreEvaluator,
            RougeType,
            SimilarityEvaluator,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Azure AI Evaluation SDK is not installed. "
            "Install the optional Foundry dependencies with "
            "`uv sync --python 3.13 --extra dev --extra foundry`."
        ) from exc

    model_config: dict[str, Any] | None = None
    evaluator_kwargs: dict[str, Any] = {}
    if any(name.lower() in _MODEL_JUDGE_EVALUATORS for name in names):
        model_config = _required_judge_model_config(config.models)
        evaluator_kwargs = {
            "is_reasoning_model": _is_reasoning_model_judge(config.models),
        }

    evaluators: dict[str, Callable[..., Any]] = {}
    for name in names:
        key = name.lower()
        if key == "f1":
            evaluators[name] = F1ScoreEvaluator()
        elif key == "rouge":
            evaluators[name] = RougeScoreEvaluator(rouge_type=RougeType.ROUGE_1)
        elif key == "bleu":
            evaluators[name] = BleuScoreEvaluator()
        elif key == "gleu":
            evaluators[name] = GleuScoreEvaluator()
        elif key == "meteor":
            evaluators[name] = MeteorScoreEvaluator()
        elif key == "groundedness":
            evaluators[name] = GroundednessEvaluator(model_config, **evaluator_kwargs)
        elif key == "relevance":
            evaluators[name] = RelevanceEvaluator(model_config, **evaluator_kwargs)
        elif key == "retrieval":
            evaluators[name] = RetrievalEvaluator(model_config, **evaluator_kwargs)
        elif key == "response_completeness":
            evaluators[name] = ResponseCompletenessEvaluator(model_config, **evaluator_kwargs)
        elif key == "document_retrieval":
            evaluators[name] = DocumentRetrievalEvaluator()
        elif key == "qa":
            evaluators[name] = QAEvaluator(model_config, **evaluator_kwargs)
        elif key == "similarity":
            evaluators[name] = SimilarityEvaluator(model_config, **evaluator_kwargs)
        else:
            raise ValueError(
                f"unsupported managed evaluator {name!r}; "
                f"supported values are {', '.join(_SUPPORTED_MANAGED_EVALUATORS)}"
            )
    return evaluators


def _placeholder_evaluators(names: list[str]) -> dict[str, Callable[..., Any]]:
    return {name: _placeholder_evaluator for name in names}


def _placeholder_evaluator(**kwargs: Any) -> dict[str, float]:
    del kwargs
    return {}


def _managed_evaluator_config(names: list[str]) -> dict[str, dict[str, dict[str, str]]]:
    return {name: {"column_mapping": _managed_column_mapping(name)} for name in names}


def _managed_column_mapping(name: str) -> dict[str, str]:
    key = name.lower()
    if key in {"f1", "rouge", "bleu", "gleu", "meteor", "similarity"}:
        return {
            "response": "${data.response}",
            "ground_truth": "${data.ground_truth}",
        }
    if key == "groundedness":
        return {
            "query": "${data.query}",
            "response": "${data.response}",
            "context": "${data.context}",
        }
    if key == "relevance":
        return {
            "query": "${data.query}",
            "response": "${data.response}",
        }
    if key == "retrieval":
        return {
            "query": "${data.query}",
            "context": "${data.context}",
        }
    if key == "response_completeness":
        return {
            "response": "${data.response}",
            "ground_truth": "${data.ground_truth}",
        }
    if key == "document_retrieval":
        return {
            "retrieval_ground_truth": "${data.retrieval_ground_truth}",
            "retrieved_documents": "${data.retrieved_documents}",
        }
    if key == "qa":
        return {
            "query": "${data.query}",
            "response": "${data.response}",
            "context": "${data.context}",
            "ground_truth": "${data.ground_truth}",
        }
    raise ValueError(
        f"unsupported managed evaluator {name!r}; "
        f"supported values are {', '.join(_SUPPORTED_MANAGED_EVALUATORS)}"
    )


def _judge_model_config_plan(
    config: ModelConfig,
    *,
    evaluator_names: list[str],
) -> dict[str, Any] | None:
    if not any(name.lower() in _MODEL_JUDGE_EVALUATORS for name in evaluator_names):
        return None

    deployment = _resolved_string(config.judge_deployment)
    endpoint = _resolved_azure_openai_endpoint(config.generator_base_url)
    api_key_configured = (
        bool(os.environ.get(config.generator_api_key_env))
        if config.generator_auth_mode == "api_key"
        else None
    )
    return {
        "required": True,
        "azure_endpoint": endpoint,
        "deployment": deployment,
        "auth_mode": config.generator_auth_mode,
        "api_key_env": config.generator_api_key_env
        if config.generator_auth_mode == "api_key"
        else None,
        "api_key_configured": api_key_configured,
        "api_version": _evaluation_api_version(),
        "is_reasoning_model": _is_reasoning_model_judge(config),
        "missing": _missing_judge_model_values(config),
    }


def _required_judge_model_config(config: ModelConfig) -> dict[str, Any]:
    missing = _missing_judge_model_values(config)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Foundry managed RAG evaluators require a judge model configuration. "
            f"Missing or unresolved values: {joined}. Configure them in .env."
        )

    model_config: dict[str, Any] = {
        "type": "azure_openai",
        "azure_deployment": _resolved_string(config.judge_deployment),
        "azure_endpoint": _resolved_azure_openai_endpoint(config.generator_base_url),
    }
    api_version = _evaluation_api_version()
    if api_version is not None:
        model_config["api_version"] = api_version

    if config.generator_auth_mode == "api_key":
        api_key = os.environ.get(config.generator_api_key_env)
        if api_key is None:
            raise RuntimeError(
                "Foundry managed RAG evaluators require an API key for api_key auth. "
                f"Set {config.generator_api_key_env} in .env."
            )
        model_config["api_key"] = api_key
    else:
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError as exc:
            raise RuntimeError(
                "Azure identity is required for azure_default_credential auth. "
                "Install the optional Foundry dependencies with "
                "`uv sync --python 3.13 --extra dev --extra foundry`."
            ) from exc
        model_config["credential"] = DefaultAzureCredential()

    return model_config


def _missing_judge_model_values(config: ModelConfig) -> list[str]:
    missing = []
    if _resolved_azure_openai_endpoint(config.generator_base_url) is None:
        missing.append("models.generator_base_url")
    if _resolved_string(config.judge_deployment) is None:
        missing.append("models.judge_deployment")
    if config.generator_auth_mode == "api_key" and not os.environ.get(config.generator_api_key_env):
        missing.append(config.generator_api_key_env)
    return missing


def _resolved_azure_openai_endpoint(base_url: str | None) -> str | None:
    resolved = _resolved_string(base_url)
    if resolved is None:
        return None

    split = urlsplit(resolved.strip())
    if not split.scheme or not split.netloc:
        return None

    path = split.path.rstrip("/")
    openai_index = path.lower().find("/openai")
    if openai_index >= 0:
        path = path[:openai_index].rstrip("/")
    return urlunsplit((split.scheme, split.netloc, path, "", ""))


def _resolved_string(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or not _is_resolved_value(stripped):
        return None
    return stripped


def _evaluation_api_version() -> str | None:
    return os.environ.get("FOUNDRY_EVALUATION_API_VERSION") or os.environ.get(
        "AZURE_OPENAI_API_VERSION"
    )


def _evaluation_name_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-") or "evaluator"


def _is_reasoning_model_judge(config: ModelConfig) -> bool:
    override = os.environ.get("FOUNDRY_EVALUATION_REASONING_MODEL")
    if override is not None:
        normalized = override.strip().lower()
        if normalized in _TRUTHY_ENV_VALUES:
            return True
        if normalized in _FALSEY_ENV_VALUES:
            return False
        raise RuntimeError(
            "FOUNDRY_EVALUATION_REASONING_MODEL must be true or false when set."
        )

    deployment = _resolved_string(config.judge_deployment)
    if deployment is None:
        return False
    return _is_reasoning_model_deployment(deployment)


def _is_reasoning_model_deployment(deployment: str) -> bool:
    return _REASONING_MODEL_DEPLOYMENT_PATTERN.search(deployment) is not None


def _azure_ai_project(config: FoundryEvaluationConfig) -> str | None:
    configured = config.azure_ai_project
    if configured and _is_resolved_value(configured):
        return configured

    for name in ("FOUNDRY_EVALUATION_PROJECT_ENDPOINT", "AZURE_AI_PROJECT"):
        value = os.environ.get(name)
        if value and _is_resolved_value(value):
            return value
    return None


def _evaluation_tags(
    config: ExperimentConfig,
    foundry_config: FoundryEvaluationConfig,
) -> dict[str, str]:
    tags = {
        "experiment_id": config.experiment.id,
        "dataset_adapter": config.dataset.adapter,
        "systems": ",".join(config.systems),
    }
    tags.update(foundry_config.tags)
    return tags


def _normalize_result(
    result: Any,
    *,
    evaluation_name: str,
    dataset_path: Path,
    result_path: Path,
    azure_ai_project: str | None,
) -> FoundryManagedEvaluationResult:
    metrics = _result_value(result, "metrics", {})
    rows = _result_value(result, "rows", [])
    oai_eval_run_ids = _result_value(result, "oai_eval_run_ids", [])
    studio_url = _result_value(result, "studio_url", None)
    return FoundryManagedEvaluationResult(
        evaluation_name=evaluation_name,
        dataset_path=str(dataset_path),
        result_path=str(result_path),
        azure_ai_project=azure_ai_project,
        metrics=dict(metrics) if isinstance(metrics, Mapping) else {},
        row_count=len(rows) if isinstance(rows, list) else 0,
        studio_url=str(studio_url) if studio_url else None,
        oai_eval_run_ids=_normalize_oai_eval_run_ids(oai_eval_run_ids),
        metadata={"sdk_result_type": result.__class__.__name__},
    )


def _normalize_oai_eval_run_ids(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if isinstance(item, Mapping):
            normalized.append({str(key): str(val) for key, val in item.items()})
    return normalized


def _result_value(result: Any, key: str, default: Any) -> Any:
    if isinstance(result, Mapping):
        return result.get(key, default)
    return getattr(result, key, default)


def _is_resolved_value(value: str) -> bool:
    return not (value.startswith("${") and value.endswith("}"))
