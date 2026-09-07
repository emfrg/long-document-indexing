from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from long_document_indexing.config import ExperimentConfig, FoundryEvaluationConfig
from long_document_indexing.storage.artifacts import ArtifactStore


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

    azure_ai_project = _azure_ai_project(foundry_config)
    evaluation_name = foundry_config.evaluation_name or config.experiment.id
    evaluators = (
        _managed_evaluators(foundry_config.managed_evaluators)
        if evaluate_fn is None
        else _placeholder_evaluators(foundry_config.managed_evaluators)
    )
    evaluator_config = _managed_evaluator_config(foundry_config.managed_evaluators)
    evaluate = evaluate_fn or _load_evaluate()
    result = evaluate(
        data=str(dataset_path),
        evaluators=evaluators,
        evaluation_name=evaluation_name,
        evaluator_config=evaluator_config,
        azure_ai_project=azure_ai_project,
        fail_on_evaluator_errors=foundry_config.fail_on_evaluator_errors,
        tags=_evaluation_tags(config, foundry_config),
    )
    normalized = _normalize_result(
        result,
        evaluation_name=evaluation_name,
        dataset_path=dataset_path,
        result_path=result_path,
        azure_ai_project=azure_ai_project,
    )
    store.write_json(foundry_config.result_path, normalized)
    return normalized


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
    return {
        "evaluation_name": foundry_config.evaluation_name or config.experiment.id,
        "dataset_path": str(dataset_path),
        "dataset_exists": dataset_path.exists(),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "result_path": str(store.experiment_dir / foundry_config.result_path),
        "azure_ai_project": azure_ai_project,
        "managed_evaluators": foundry_config.managed_evaluators,
        "evaluator_config": _managed_evaluator_config(foundry_config.managed_evaluators),
        "fail_on_evaluator_errors": foundry_config.fail_on_evaluator_errors,
        "tags": _evaluation_tags(config, foundry_config),
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


def _managed_evaluators(names: list[str]) -> dict[str, Callable[..., Any]]:
    try:
        from azure.ai.evaluation import (
            BleuScoreEvaluator,
            F1ScoreEvaluator,
            GleuScoreEvaluator,
            MeteorScoreEvaluator,
            RougeScoreEvaluator,
            RougeType,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Azure AI Evaluation SDK is not installed. "
            "Install the optional Foundry dependencies with "
            "`uv sync --python 3.13 --extra dev --extra foundry`."
        ) from exc

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
        else:
            raise ValueError(
                f"unsupported managed evaluator {name!r}; "
                "supported values are f1, rouge, bleu, gleu, and meteor"
            )
    return evaluators


def _placeholder_evaluators(names: list[str]) -> dict[str, Callable[..., Any]]:
    return {name: _placeholder_evaluator for name in names}


def _placeholder_evaluator(**kwargs: Any) -> dict[str, float]:
    del kwargs
    return {}


def _managed_evaluator_config(names: list[str]) -> dict[str, dict[str, dict[str, str]]]:
    return {
        name: {
            "column_mapping": {
                "response": "${data.response}",
                "ground_truth": "${data.ground_truth}",
            }
        }
        for name in names
    }


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
