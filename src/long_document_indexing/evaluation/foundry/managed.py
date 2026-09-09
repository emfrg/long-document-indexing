from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

from long_document_indexing.config import ExperimentConfig, FoundryEvaluationConfig, ModelConfig
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
        _managed_evaluators(foundry_config.managed_evaluators, config=config)
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
        "judge_model_config": _judge_model_config_plan(
            config.models,
            evaluator_names=foundry_config.managed_evaluators,
        ),
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
    if any(name.lower() in _MODEL_JUDGE_EVALUATORS for name in names):
        model_config = _required_judge_model_config(config.models)

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
            evaluators[name] = GroundednessEvaluator(model_config)
        elif key == "relevance":
            evaluators[name] = RelevanceEvaluator(model_config)
        elif key == "retrieval":
            evaluators[name] = RetrievalEvaluator(model_config)
        elif key == "response_completeness":
            evaluators[name] = ResponseCompletenessEvaluator(model_config)
        elif key == "document_retrieval":
            evaluators[name] = DocumentRetrievalEvaluator()
        elif key == "qa":
            evaluators[name] = QAEvaluator(model_config)
        elif key == "similarity":
            evaluators[name] = SimilarityEvaluator(model_config)
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

    deployment = _resolved_string(config.judge_deployment or config.generator_deployment)
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
        "azure_deployment": _resolved_string(
            config.judge_deployment or config.generator_deployment
        ),
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
    if _resolved_string(config.judge_deployment or config.generator_deployment) is None:
        missing.append("models.judge_deployment or models.generator_deployment")
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
