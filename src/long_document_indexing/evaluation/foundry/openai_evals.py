from __future__ import annotations

import collections
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from long_document_indexing.config import ExperimentConfig, FoundryEvaluationConfig
from long_document_indexing.evaluation.foundry.managed import _azure_ai_project
from long_document_indexing.storage.artifacts import ArtifactStore

OPENAI_EVALS_DATASET_PATH = Path("evaluations/foundry/openai-evals-dataset.jsonl")
OPENAI_EVALS_RESULT_PATH = Path("evaluations/foundry/openai-evals-result.json")
OPENAI_EVALS_TOKEN_SCOPE = "https://ai.azure.com/.default"

TOKEN_F1_GRADER_SOURCE = """
import collections
import re


def _tokens(value):
    return re.findall(r"\\w+", str(value or "").lower())


def grade(sample, item):
    prediction = _tokens(item.get("response"))
    reference = _tokens(item.get("ground_truth"))
    if not prediction or not reference:
        return 0.0
    common = sum((collections.Counter(prediction) & collections.Counter(reference)).values())
    if common == 0:
        return 0.0
    precision = common / len(prediction)
    recall = common / len(reference)
    return 2 * precision * recall / (precision + recall)
""".strip()


class FoundryOpenAIEvalsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_name: str
    eval_id: str
    run_name: str
    run_id: str
    dataset_path: str
    run_dataset_path: str
    result_path: str
    project_endpoint: str
    file_id: str | None = None
    status: str
    report_url: str | None = None
    result_counts: dict[str, Any] = Field(default_factory=dict)
    score_averages: dict[str, float] = Field(default_factory=dict)
    score_ranges: dict[str, dict[str, float]] = Field(default_factory=dict)
    row_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_foundry_openai_evals_plan(
    *,
    config: ExperimentConfig,
    store: ArtifactStore,
    evaluation_name: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    foundry_config = config.evaluation.foundry
    dataset_path = store.experiment_dir / foundry_config.dataset_path
    return {
        "evaluation_name": _openai_evaluation_name(config, evaluation_name),
        "run_name": _openai_run_name(config, run_name),
        "dataset_path": str(dataset_path),
        "dataset_exists": dataset_path.exists(),
        "run_dataset_path": str(store.experiment_dir / OPENAI_EVALS_DATASET_PATH),
        "result_path": str(store.experiment_dir / OPENAI_EVALS_RESULT_PATH),
        "project_endpoint": _required_project_endpoint(foundry_config),
        "managed_evaluators": foundry_config.managed_evaluators,
        "testing_criteria": _testing_criteria(foundry_config.managed_evaluators),
        "tags": _evaluation_metadata(config, foundry_config),
    }


def run_foundry_openai_evals(
    *,
    config: ExperimentConfig,
    store: ArtifactStore,
    evaluation_name: str | None = None,
    run_name: str | None = None,
    client: Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> FoundryOpenAIEvalsResult:
    foundry_config = config.evaluation.foundry
    project_endpoint = _required_project_endpoint(foundry_config)
    dataset_path = store.experiment_dir / foundry_config.dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Foundry evaluation dataset not found: {dataset_path}. "
            "Run `ldi evaluate` or `ldi export-foundry-eval` first."
        )

    client = client or _build_openai_client(project_endpoint)
    name = _openai_evaluation_name(config, evaluation_name)
    run_name = _openai_run_name(config, run_name)
    eval_obj = _get_or_create_eval(client, config=config, name=name)
    run_dataset_path = _write_openai_evals_dataset(
        source_path=dataset_path,
        target_path=store.experiment_dir / OPENAI_EVALS_DATASET_PATH,
    )
    file_obj = _upload_file(client, run_dataset_path)
    run = client.evals.runs.create(
        _get_required(eval_obj, "id"),
        name=run_name,
        data_source={
            "type": "jsonl",
            "source": {
                "type": "file_id",
                "id": _get_required(file_obj, "id"),
            },
        },
        metadata=_evaluation_metadata(config, foundry_config),
        timeout=120,
    )
    run = _poll_run(
        client,
        eval_id=_get_required(eval_obj, "id"),
        run_id=_get_required(run, "id"),
        sleep_fn=sleep_fn,
    )
    output_items = _list_output_items(
        client,
        eval_id=_get_required(eval_obj, "id"),
        run_id=_get_required(run, "id"),
    )
    result = _normalize_result(
        config=config,
        evaluation_name=name,
        eval_obj=eval_obj,
        run_name=run_name,
        run=run,
        file_obj=file_obj,
        dataset_path=dataset_path,
        run_dataset_path=run_dataset_path,
        result_path=store.experiment_dir / OPENAI_EVALS_RESULT_PATH,
        project_endpoint=project_endpoint,
        output_items=output_items,
    )
    store.write_json(OPENAI_EVALS_RESULT_PATH, result)
    return result


def _build_openai_client(project_endpoint: str) -> Any:
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Foundry portal-visible Evals require the optional `foundry` extra. "
            "Install it with `uv sync --extra foundry`."
        ) from exc

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        OPENAI_EVALS_TOKEN_SCOPE,
    )
    return OpenAI(
        api_key=token_provider,
        base_url=_openai_evals_base_url(project_endpoint),
        timeout=120,
    )


def _get_or_create_eval(client: Any, *, config: ExperimentConfig, name: str) -> Any:
    for existing in _list_evals(client):
        if _get(existing, "name") == name:
            return existing

    foundry_config = config.evaluation.foundry
    return client.evals.create(
        name=name,
        data_source_config={
            "type": "custom",
            "item_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "response": {"type": "string"},
                    "ground_truth": {"type": "string"},
                    "system_id": {"type": "string"},
                    "item_id": {"type": "string"},
                },
                "required": ["query", "response", "ground_truth", "system_id", "item_id"],
            },
        },
        testing_criteria=_testing_criteria(foundry_config.managed_evaluators),
        metadata=_evaluation_metadata(config, foundry_config),
        timeout=120,
    )


def _list_evals(client: Any) -> list[Any]:
    items: list[Any] = []
    after: str | None = None
    while True:
        kwargs: dict[str, Any] = {"limit": 100, "order": "desc"}
        if after is not None:
            kwargs["after"] = after
        page = client.evals.list(**kwargs)
        items.extend(_get(page, "data", []))
        if not _get(page, "has_more", False):
            return items
        after = _get(page, "last_id")
        if after is None:
            return items


def _upload_file(client: Any, path: Path) -> Any:
    with path.open("rb") as handle:
        file_obj = client.files.create(file=handle, purpose="evals", timeout=300)
    wait_for_processing = getattr(client.files, "wait_for_processing", None)
    if callable(wait_for_processing):
        return wait_for_processing(
            _get_required(file_obj, "id"),
            poll_interval=2,
            max_wait_seconds=300,
        )
    return file_obj


def _poll_run(
    client: Any,
    *,
    eval_id: str,
    run_id: str,
    sleep_fn: Callable[[float], None],
) -> Any:
    terminal = {"completed", "failed", "canceled"}
    for _ in range(60):
        run = client.evals.runs.retrieve(run_id, eval_id=eval_id)
        if _get(run, "status") in terminal:
            return run
        sleep_fn(10)
    raise RuntimeError(f"Foundry eval run did not finish: {run_id}")


def _list_output_items(client: Any, *, eval_id: str, run_id: str) -> list[Any]:
    items: list[Any] = []
    after: str | None = None
    while True:
        kwargs: dict[str, Any] = {"eval_id": eval_id, "limit": 100, "order": "asc"}
        if after is not None:
            kwargs["after"] = after
        page = client.evals.runs.output_items.list(run_id, **kwargs)
        page_items = list(_get(page, "data", []))
        items.extend(page_items)
        if not _get(page, "has_more", False):
            return items
        after = _get(page, "last_id")
        if after is None:
            return items


def _write_openai_evals_dataset(*, source_path: Path, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open(encoding="utf-8") as source, target_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            target.write(json.dumps({"item": json.loads(line)}, sort_keys=True))
            target.write("\n")
    return target_path


def _testing_criteria(names: list[str]) -> list[dict[str, Any]]:
    criteria: list[dict[str, Any]] = []
    for name in names:
        key = name.lower()
        if key == "f1":
            criteria.append(
                {
                    "type": "python",
                    "name": "token_f1",
                    "source": TOKEN_F1_GRADER_SOURCE,
                    "image_tag": "2025-05-08",
                    "pass_threshold": 0.5,
                }
            )
        elif key == "rouge":
            criteria.append(
                {
                    "type": "text_similarity",
                    "name": "rouge_1",
                    "input": "{{item.response}}",
                    "reference": "{{item.ground_truth}}",
                    "evaluation_metric": "rouge_1",
                    "pass_threshold": 0.5,
                }
            )
        elif key in {"bleu", "gleu", "meteor"}:
            criteria.append(
                {
                    "type": "text_similarity",
                    "name": key,
                    "input": "{{item.response}}",
                    "reference": "{{item.ground_truth}}",
                    "evaluation_metric": key,
                    "pass_threshold": 0.5,
                }
            )
        else:
            raise ValueError(
                f"unsupported portal-visible Foundry evaluator {name!r}; "
                "supported values are f1, rouge, bleu, gleu, and meteor"
            )
    return criteria


def _normalize_result(
    *,
    config: ExperimentConfig,
    evaluation_name: str,
    eval_obj: Any,
    run_name: str,
    run: Any,
    file_obj: Any,
    dataset_path: Path,
    run_dataset_path: Path,
    result_path: Path,
    project_endpoint: str,
    output_items: list[Any],
) -> FoundryOpenAIEvalsResult:
    averages, ranges = _score_summary(output_items)
    return FoundryOpenAIEvalsResult(
        evaluation_name=evaluation_name,
        eval_id=_get_required(eval_obj, "id"),
        run_name=run_name,
        run_id=_get_required(run, "id"),
        dataset_path=str(dataset_path),
        run_dataset_path=str(run_dataset_path),
        result_path=str(result_path),
        project_endpoint=project_endpoint,
        file_id=_get(file_obj, "id"),
        status=str(_get(run, "status")),
        report_url=_get(run, "report_url"),
        result_counts=_dump_model(_get(run, "result_counts", {})),
        score_averages=averages,
        score_ranges=ranges,
        row_count=len(output_items),
        metadata={
            "experiment_id": config.experiment.id,
            "evaluator_path": "openai_evals",
        },
    )


def _score_summary(output_items: list[Any]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    values_by_name: dict[str, list[float]] = collections.defaultdict(list)
    for item in output_items:
        for result in _get(item, "results", []):
            score = _get(result, "score")
            if not isinstance(score, int | float):
                continue
            values_by_name[str(_get(result, "name"))].append(float(score))
    averages = {
        name: sum(values) / len(values) for name, values in sorted(values_by_name.items())
    }
    ranges = {
        name: {"min": min(values), "max": max(values)}
        for name, values in sorted(values_by_name.items())
    }
    return averages, ranges


def _openai_evaluation_name(config: ExperimentConfig, configured: str | None) -> str:
    base = configured or config.evaluation.foundry.evaluation_name or config.experiment.id
    return base


def _openai_run_name(config: ExperimentConfig, configured: str | None) -> str:
    return configured or f"{config.experiment.id}-precomputed-run"


def _evaluation_metadata(
    config: ExperimentConfig,
    foundry_config: FoundryEvaluationConfig,
) -> dict[str, str]:
    metadata = {
        "experiment_id": config.experiment.id,
        "dataset_adapter": config.dataset.adapter,
        "systems": ",".join(config.systems),
        "trigger_type": "oneoff",
    }
    metadata.update(foundry_config.tags)
    return metadata


def _required_project_endpoint(config: FoundryEvaluationConfig) -> str:
    project_endpoint = _azure_ai_project(config)
    if project_endpoint is None:
        raise RuntimeError(
            "Foundry portal-visible Evals require a project endpoint. "
            "Set FOUNDRY_EVALUATION_PROJECT_ENDPOINT in .env."
        )
    return project_endpoint


def _openai_evals_base_url(project_endpoint: str) -> str:
    normalized = project_endpoint.strip().rstrip("/")
    if normalized.endswith("/openai/v1"):
        return f"{normalized}/"
    return f"{normalized}/openai/v1/"


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _get_required(value: Any, key: str) -> str:
    resolved = _get(value, key)
    if not isinstance(resolved, str) or not resolved:
        raise RuntimeError(f"Foundry API response did not include required field {key!r}")
    return resolved


def _dump_model(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return {}
