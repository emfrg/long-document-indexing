from __future__ import annotations

import collections
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from long_document_indexing.config import ExperimentConfig, FoundryEvaluationConfig
from long_document_indexing.evaluation.foundry.adapter import foundry_item_schema
from long_document_indexing.evaluation.foundry.managed import _azure_ai_project
from long_document_indexing.storage.artifacts import ArtifactStore, atomic_write_text

OPENAI_EVALS_DATASET_PATH = Path("evaluations/foundry/openai-evals-dataset.jsonl")
OPENAI_EVALS_RESULT_PATH = Path("evaluations/foundry/openai-evals-result.json")
OPENAI_EVALS_PENDING_PATH = Path("evaluations/foundry/openai-evals-pending.json")
OPENAI_EVALS_SYSTEM_RESULTS_PATH = Path(
    "evaluations/foundry/openai-evals-system-results.json"
)
OPENAI_EVALS_TOKEN_SCOPE = "https://ai.azure.com/.default"
OPENAI_EVALS_MAX_ATTEMPTS = 3
OPENAI_EVALS_RETRY_DELAY_SECONDS = 10.0
_TRANSIENT_ERROR_MARKERS = (
    "429",
    "apiconnectionerror",
    "apitimeouterror",
    "connection error",
    "connection reset",
    "rate limit",
    "readerror",
    "readtimeout",
    "temporarily unavailable",
    "timeout",
)


class FoundryOpenAIEvalsIncompleteError(RuntimeError):
    """Raised when a portal-visible evaluation run is not fully complete."""

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

DOCUMENT_RETRIEVAL_PRECISION_GRADER_SOURCE = """
def _document_ids(values):
    ids = []
    for value in values or []:
        if isinstance(value, dict):
            document_id = value.get("document_id")
        else:
            document_id = None
        if document_id and document_id not in ids:
            ids.append(document_id)
    return ids


def grade(sample, item):
    retrieved = _document_ids(item.get("retrieved_documents"))
    relevant = set(_document_ids(item.get("retrieval_ground_truth")))
    if not retrieved:
        return 0.0
    if not relevant:
        relevant = set(item.get("relevant_document_ids") or [])
    if not relevant:
        return 0.0
    return len(set(retrieved) & relevant) / len(retrieved)
""".strip()

DOCUMENT_RETRIEVAL_RECALL_GRADER_SOURCE = """
def _document_ids(values):
    ids = []
    for value in values or []:
        if isinstance(value, dict):
            document_id = value.get("document_id")
        else:
            document_id = None
        if document_id and document_id not in ids:
            ids.append(document_id)
    return ids


def grade(sample, item):
    retrieved = set(_document_ids(item.get("retrieved_documents")))
    relevant = set(_document_ids(item.get("retrieval_ground_truth")))
    if not relevant:
        relevant = set(item.get("relevant_document_ids") or [])
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)
""".strip()

CONTEXT_PRECISION_AT_4_GRADER_SOURCE = """
def _relevant_ids(item):
    segment_ids = set(item.get("relevant_segment_ids") or [])
    if segment_ids:
        return segment_ids, "segment_id"
    return set(item.get("relevant_document_ids") or []), "document_id"


def grade(sample, item):
    relevant, id_key = _relevant_ids(item)
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    seen = set()
    contexts = sorted(item.get("retrieved_context") or [], key=lambda row: row.get("rank", 0))
    for rank, context in enumerate(contexts[:4], start=1):
        context_id = context.get(id_key)
        if not context_id or context_id in seen:
            continue
        seen.add(context_id)
        if context_id not in relevant:
            continue
        hits += 1
        precision_sum += hits / rank
    if hits == 0:
        return 0.0
    return precision_sum / hits
""".strip()

CONTEXT_RECALL_AT_4_GRADER_SOURCE = """
def _relevant_ids(item):
    segment_ids = set(item.get("relevant_segment_ids") or [])
    if segment_ids:
        return segment_ids, "segment_id"
    return set(item.get("relevant_document_ids") or []), "document_id"


def grade(sample, item):
    relevant, id_key = _relevant_ids(item)
    if not relevant:
        return 0.0
    contexts = sorted(item.get("retrieved_context") or [], key=lambda row: row.get("rank", 0))
    retrieved = {
        context.get(id_key)
        for context in contexts[:4]
        if context.get(id_key)
    }
    return len(retrieved & relevant) / len(relevant)
""".strip()

EVIDENCE_QUOTE_RECALL_AT_4_GRADER_SOURCE = """
import re


def _normalize(value):
    return re.sub(r"\\s+", " ", str(value or "")).strip().lower()


def grade(sample, item):
    evidence = (item.get("expected_behavior") or {}).get("evidence") or []
    quotes = [_normalize(span.get("quote")) for span in evidence if span.get("quote")]
    if not quotes:
        return 0.0
    contexts = sorted(item.get("retrieved_context") or [], key=lambda row: row.get("rank", 0))
    texts = [_normalize(context.get("text")) for context in contexts[:4]]
    hits = sum(1 for quote in quotes if any(quote in text for text in texts))
    return hits / len(quotes)
""".strip()

CITATION_SUPPORT_RATE_GRADER_SOURCE = """
import re


def _normalize(value):
    return re.sub(r"\\s+", " ", str(value or "")).strip().lower()


def grade(sample, item):
    citations = item.get("citations") or []
    if not citations:
        return 0.0
    contexts = item.get("retrieved_context") or []
    supported = 0
    for citation in citations:
        quote = _normalize(citation.get("quote"))
        if not quote:
            continue
        for context in contexts:
            if context.get("document_id") != citation.get("document_id"):
                continue
            citation_segment = citation.get("segment_id")
            if citation_segment and context.get("segment_id") != citation_segment:
                continue
            if quote in _normalize(context.get("text")):
                supported += 1
                break
    return supported / len(citations)
""".strip()

ANSWER_REFERENCE_TOKEN_RECALL_GRADER_SOURCE = """
import collections
import re


def _tokens(value):
    return re.findall(r"\\w+", str(value or "").lower())


def grade(sample, item):
    prediction = collections.Counter(_tokens(item.get("response")))
    reference = collections.Counter(_tokens(item.get("ground_truth")))
    reference_count = sum(reference.values())
    if reference_count == 0:
        return 0.0
    overlap = sum(
        min(prediction[token], reference[token])
        for token in prediction.keys() & reference.keys()
    )
    return overlap / reference_count
""".strip()

ANSWER_REFERENCE_TOKEN_F1_GRADER_SOURCE = """
import collections
import re


def _tokens(value):
    return re.findall(r"\\w+", str(value or "").lower())


def grade(sample, item):
    prediction = collections.Counter(_tokens(item.get("response")))
    reference = collections.Counter(_tokens(item.get("ground_truth")))
    prediction_count = sum(prediction.values())
    reference_count = sum(reference.values())
    if prediction_count == 0 or reference_count == 0:
        return 0.0
    overlap = sum(
        min(prediction[token], reference[token])
        for token in prediction.keys() & reference.keys()
    )
    precision = overlap / prediction_count
    recall = overlap / reference_count
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
""".strip()


class FoundryOpenAIEvalsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_name: str
    eval_id: str
    run_name: str
    run_id: str
    system_id: str
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
    score_counts: dict[str, int] = Field(default_factory=dict)
    row_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class FoundryOpenAIEvalsBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_name: str
    result_path: str
    project_endpoint: str
    mode: str
    row_count: int = 0
    systems: list[str] = Field(default_factory=list)
    results: list[FoundryOpenAIEvalsResult] = Field(default_factory=list)


def build_foundry_openai_evals_plan(
    *,
    config: ExperimentConfig,
    store: ArtifactStore,
    evaluation_name: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    foundry_config = config.evaluation.foundry
    dataset_path = store.experiment_dir / foundry_config.dataset_path
    systems = _dataset_system_ids(dataset_path, configured_systems=config.systems)
    return {
        "evaluation_name": _openai_evaluation_name(config, evaluation_name),
        "mode": "system_comparison",
        "dataset_path": str(dataset_path),
        "dataset_exists": dataset_path.exists(),
        "result_path": str(store.experiment_dir / OPENAI_EVALS_SYSTEM_RESULTS_PATH),
        "systems": systems,
        "runs": [
            {
                "system_id": system_id,
                "run_name": _openai_system_run_name(config, run_name, system_id),
                "run_dataset_path": str(
                    store.experiment_dir
                    / _system_artifact_path(OPENAI_EVALS_DATASET_PATH, system_id)
                ),
                "result_path": str(
                    store.experiment_dir
                    / _system_artifact_path(OPENAI_EVALS_RESULT_PATH, system_id)
                ),
            }
            for system_id in systems
        ],
        "project_endpoint": _required_project_endpoint(foundry_config),
        "managed_evaluators": foundry_config.managed_evaluators,
        "testing_criteria": _testing_criteria(foundry_config.managed_evaluators),
        "tags": _evaluation_metadata(config, foundry_config),
    }


def run_foundry_openai_evals_for_system(
    *,
    config: ExperimentConfig,
    store: ArtifactStore,
    system_id: str,
    evaluation_name: str | None = None,
    run_name: str | None = None,
    client: Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    status_callback: Callable[[str], None] | None = None,
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
    run_name = run_name or _openai_system_run_name(config, None, system_id)
    _emit_status(status_callback, f"Preparing evaluation: {name}")
    run_dataset_artifact_path = _system_artifact_path(
        OPENAI_EVALS_DATASET_PATH, system_id
    )
    result_artifact_path = _system_artifact_path(OPENAI_EVALS_RESULT_PATH, system_id)
    pending_artifact_path = _system_artifact_path(OPENAI_EVALS_PENDING_PATH, system_id)
    _emit_status(status_callback, f"Preparing upload dataset: {run_dataset_artifact_path}")
    run_dataset_path = _write_openai_evals_dataset(
        source_path=dataset_path,
        target_path=store.experiment_dir / run_dataset_artifact_path,
        system_id=system_id,
    )
    expected_row_count = _jsonl_row_count(run_dataset_path)
    publish_signature = _openai_evals_publish_signature(
        config=config,
        evaluation_name=name,
        run_name=run_name,
        system_id=system_id,
        run_dataset_path=run_dataset_path,
    )
    pending = _load_pending_openai_eval_run(
        store=store,
        pending_artifact_path=pending_artifact_path,
        publish_signature=publish_signature,
    )
    resumed_pending = pending is not None
    if pending is not None:
        eval_id = pending["eval_id"]
        run_id = pending["run_id"]
        file_id = pending["file_id"]
        eval_obj = {"id": eval_id}
        file_obj = {"id": file_id}
        _emit_status(status_callback, f"Resuming existing run: {run_id}")
    else:
        eval_obj = _get_or_create_eval(client, config=config, name=name)
        eval_id = _get_required(eval_obj, "id")
        _emit_status(status_callback, "Uploading dataset and waiting for file processing")
        file_obj = _upload_file(client, run_dataset_path, status_callback=status_callback)
        file_id = _get_required(file_obj, "id")
        _emit_status(status_callback, f"Creating run: {run_name}")
        run = client.evals.runs.create(
            eval_id,
            name=run_name,
            data_source={
                "type": "jsonl",
                "source": {
                    "type": "file_id",
                    "id": file_id,
                },
            },
            metadata=_evaluation_metadata(config, foundry_config, system_id=system_id),
            timeout=120,
        )
        run_id = _get_required(run, "id")
        store.write_json(
            pending_artifact_path,
            {
                "eval_id": eval_id,
                "file_id": file_id,
                "publish_signature": publish_signature,
                "run_id": run_id,
            },
        )
    _emit_status(status_callback, f"Waiting for run to finish: {run_id}")
    try:
        run = _poll_run(
            client,
            eval_id=eval_id,
            run_id=run_id,
            sleep_fn=sleep_fn,
            status_callback=status_callback,
        )
    except Exception as exc:
        if resumed_pending and _is_missing_remote_run_error(exc):
            (store.experiment_dir / pending_artifact_path).unlink(missing_ok=True)
            raise FoundryOpenAIEvalsIncompleteError(
                f"Saved Foundry eval run {run_id} no longer exists"
            ) from exc
        raise
    try:
        _require_completed_openai_eval_run(run, expected_row_count=expected_row_count)
    except FoundryOpenAIEvalsIncompleteError:
        if _openai_eval_run_requires_replacement(run):
            (store.experiment_dir / pending_artifact_path).unlink(missing_ok=True)
        raise
    _emit_status(status_callback, "Fetching output items")
    output_items = _list_output_items(
        client,
        eval_id=eval_id,
        run_id=run_id,
    )
    if len(output_items) != expected_row_count:
        raise FoundryOpenAIEvalsIncompleteError(
            f"Foundry eval run {run_id} returned {len(output_items)} of "
            f"{expected_row_count} output items"
        )
    _validate_openai_evals_output_items(
        output_items,
        expected_score_names=_testing_criteria_names(foundry_config),
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
        result_path=store.experiment_dir / result_artifact_path,
        project_endpoint=project_endpoint,
        output_items=output_items,
        system_id=system_id,
    )
    result.metadata["publish_signature"] = publish_signature
    result.metadata["completion_status"] = "complete"
    _emit_status(status_callback, f"Writing result: {result_artifact_path}")
    store.write_json(result_artifact_path, result)
    (store.experiment_dir / pending_artifact_path).unlink(missing_ok=True)
    return result


def run_foundry_openai_evals_per_system(
    *,
    config: ExperimentConfig,
    store: ArtifactStore,
    evaluation_name: str | None = None,
    run_name: str | None = None,
    client: Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    status_callback: Callable[[str], None] | None = None,
) -> FoundryOpenAIEvalsBatchResult:
    foundry_config = config.evaluation.foundry
    project_endpoint = _required_project_endpoint(foundry_config)
    dataset_path = store.experiment_dir / foundry_config.dataset_path
    systems = _dataset_system_ids(dataset_path, configured_systems=config.systems)
    if not systems:
        raise RuntimeError(
            "Foundry per-system publish requires exported rows with a system_id field."
        )

    client = client or _build_openai_client(project_endpoint)
    name = _openai_evaluation_name(config, evaluation_name)
    results: list[FoundryOpenAIEvalsResult] = []
    for index, system_id in enumerate(systems, start=1):
        system_run_name = _openai_system_run_name(config, run_name, system_id)
        result_artifact_path = _system_artifact_path(OPENAI_EVALS_RESULT_PATH, system_id)
        _emit_status(
            status_callback,
            f"Publishing system {index}/{len(systems)}: {system_id}",
        )
        reusable = _load_reusable_openai_evals_result(
            config=config,
            store=store,
            result_artifact_path=result_artifact_path,
            dataset_path=dataset_path,
            evaluation_name=name,
            run_name=system_run_name,
            system_id=system_id,
        )
        if reusable is not None:
            _emit_status(status_callback, f"Reusing completed system: {system_id}")
            results.append(reusable)
            continue

        result = _run_openai_evals_system_with_retries(
            config=config,
            store=store,
            system_id=system_id,
            evaluation_name=name,
            run_name=system_run_name,
            client=client,
            sleep_fn=sleep_fn,
            status_callback=status_callback,
        )
        results.append(result)

    _validate_openai_evals_batch(
        results,
        config=config,
        systems=systems,
        dataset_path=dataset_path,
    )

    batch_result = FoundryOpenAIEvalsBatchResult(
        evaluation_name=name,
        result_path=str(store.experiment_dir / OPENAI_EVALS_SYSTEM_RESULTS_PATH),
        project_endpoint=project_endpoint,
        mode="system_comparison",
        row_count=sum(result.row_count for result in results),
        systems=systems,
        results=results,
    )
    store.write_json(OPENAI_EVALS_SYSTEM_RESULTS_PATH, batch_result)
    return batch_result


def _run_openai_evals_system_with_retries(
    *,
    config: ExperimentConfig,
    store: ArtifactStore,
    system_id: str,
    evaluation_name: str,
    run_name: str,
    client: Any,
    sleep_fn: Callable[[float], None],
    status_callback: Callable[[str], None] | None,
) -> FoundryOpenAIEvalsResult:
    for attempt in range(1, OPENAI_EVALS_MAX_ATTEMPTS + 1):
        try:
            return run_foundry_openai_evals_for_system(
                config=config,
                store=store,
                system_id=system_id,
                evaluation_name=evaluation_name,
                run_name=run_name,
                client=client,
                sleep_fn=sleep_fn,
                status_callback=status_callback,
            )
        except Exception as exc:
            can_retry = attempt < OPENAI_EVALS_MAX_ATTEMPTS and (
                isinstance(exc, FoundryOpenAIEvalsIncompleteError)
                or _is_transient_openai_evals_error(exc)
            )
            if not can_retry:
                raise RuntimeError(
                    f"Foundry publish for system {system_id!r} did not complete after "
                    f"{attempt} attempt(s): {exc}"
                ) from exc
            delay = OPENAI_EVALS_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
            _emit_status(
                status_callback,
                f"Retrying system {system_id} after {delay:g}s "
                f"(attempt {attempt + 1}/{OPENAI_EVALS_MAX_ATTEMPTS})",
            )
            sleep_fn(delay)
    raise AssertionError("Foundry publish retry loop terminated unexpectedly")


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
        max_retries=5,
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
            "item_schema": foundry_item_schema(),
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


def _upload_file(
    client: Any,
    path: Path,
    *,
    status_callback: Callable[[str], None] | None = None,
) -> Any:
    with path.open("rb") as handle:
        file_obj = client.files.create(file=handle, purpose="evals", timeout=300)
    wait_for_processing = getattr(client.files, "wait_for_processing", None)
    if callable(wait_for_processing):
        _emit_status(
            status_callback,
            f"Waiting for uploaded file to process: {_get_required(file_obj, 'id')}",
        )
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
    status_callback: Callable[[str], None] | None = None,
) -> Any:
    terminal = {"completed", "failed", "canceled"}
    last_status: str | None = None
    last_report = 0.0
    for _ in range(60):
        run = client.evals.runs.retrieve(run_id, eval_id=eval_id)
        status = str(_get(run, "status"))
        now = time.monotonic()
        should_report = status != last_status or now - last_report >= 60
        if should_report:
            _emit_status(status_callback, f"Run status: {status}")
            last_status = status
            last_report = now
        if status in terminal:
            return run
        sleep_fn(10)
    raise RuntimeError(f"Foundry eval run did not finish: {run_id}")


def _require_completed_openai_eval_run(run: Any, *, expected_row_count: int) -> None:
    run_id = _get_required(run, "id")
    status = str(_get(run, "status", "")).lower()
    if status != "completed":
        raise FoundryOpenAIEvalsIncompleteError(
            f"Foundry eval run {run_id} ended with status {status!r}"
        )
    counts = _dump_model(_get(run, "result_counts", {}))
    total = int(counts.get("total", 0))
    errored = int(counts.get("errored", 0))
    if total != expected_row_count or errored != 0:
        raise FoundryOpenAIEvalsIncompleteError(
            f"Foundry eval run {run_id} expected {expected_row_count} completed rows; "
            f"reported total={total}, errored={errored}"
        )


def _emit_status(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


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


def _write_openai_evals_dataset(
    *,
    source_path: Path,
    target_path: Path,
    system_id: str | None = None,
) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    content, row_count = _openai_evals_dataset_content(source_path, system_id=system_id)
    atomic_write_text(target_path, content)
    if system_id is not None and row_count == 0:
        raise RuntimeError(
            f"Foundry per-system publish found no exported rows for system {system_id!r}."
        )
    return target_path


def _openai_evals_dataset_content(
    source_path: Path,
    *,
    system_id: str | None,
) -> tuple[str, int]:
    lines = []
    with source_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if system_id is not None and row.get("system_id") != system_id:
                continue
            lines.append(json.dumps({"item": row}, sort_keys=True))
    return "\n".join(lines) + ("\n" if lines else ""), len(lines)


def _dataset_system_ids(source_path: Path, *, configured_systems: list[str]) -> list[str]:
    if not source_path.exists():
        return []
    observed: set[str] = set()
    with source_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            system_id = row.get("system_id")
            if isinstance(system_id, str) and system_id:
                observed.add(system_id)
    ordered = [system_id for system_id in configured_systems if system_id in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def _system_artifact_path(path: Path, system_id: str | None) -> Path:
    if system_id is None:
        return path
    safe_system_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in system_id
    ).strip("-")
    if not safe_system_id:
        raise RuntimeError(f"invalid system id for Foundry artifact path: {system_id!r}")
    return path.with_name(f"{path.stem}-{safe_system_id}{path.suffix}")


def _load_reusable_openai_evals_result(
    *,
    config: ExperimentConfig,
    store: ArtifactStore,
    result_artifact_path: Path,
    dataset_path: Path,
    evaluation_name: str,
    run_name: str,
    system_id: str,
) -> FoundryOpenAIEvalsResult | None:
    path = store.experiment_dir / result_artifact_path
    if not path.exists():
        return None
    try:
        result = FoundryOpenAIEvalsResult.model_validate(store.read_json(result_artifact_path))
    except (OSError, ValueError):
        return None
    if (
        result.evaluation_name != evaluation_name
        or result.run_name != run_name
        or result.system_id != system_id
        or result.status.lower() != "completed"
        or result.metadata.get("completion_status") not in {None, "complete"}
    ):
        return None

    expected_content, expected_row_count = _openai_evals_dataset_content(
        dataset_path,
        system_id=system_id,
    )
    run_dataset_path = Path(result.run_dataset_path)
    if not run_dataset_path.exists():
        return None
    if run_dataset_path.read_text(encoding="utf-8") != expected_content:
        return None
    expected_signature = _openai_evals_publish_signature(
        config=config,
        evaluation_name=evaluation_name,
        run_name=run_name,
        system_id=system_id,
        run_dataset_path=run_dataset_path,
    )
    recorded_signature = result.metadata.get("publish_signature")
    if recorded_signature is not None and recorded_signature != expected_signature:
        return None
    try:
        _validate_completed_openai_evals_result(
            result,
            expected_row_count=expected_row_count,
            expected_score_names=_testing_criteria_names(config.evaluation.foundry),
        )
    except FoundryOpenAIEvalsIncompleteError:
        return None
    return result


def _load_pending_openai_eval_run(
    *,
    store: ArtifactStore,
    pending_artifact_path: Path,
    publish_signature: str,
) -> dict[str, str] | None:
    path = store.experiment_dir / pending_artifact_path
    if not path.exists():
        return None
    try:
        pending = store.read_json(pending_artifact_path)
    except (OSError, ValueError):
        return None
    if pending.get("publish_signature") != publish_signature:
        return None
    required = {key: pending.get(key) for key in ("eval_id", "file_id", "run_id")}
    if not all(isinstance(value, str) and value for value in required.values()):
        return None
    return {key: str(value) for key, value in required.items()}


def _validate_completed_openai_evals_result(
    result: FoundryOpenAIEvalsResult,
    *,
    expected_row_count: int,
    expected_score_names: set[str],
) -> None:
    counts = result.result_counts
    total = int(counts.get("total", 0))
    errored = int(counts.get("errored", 0))
    if result.status.lower() != "completed":
        raise FoundryOpenAIEvalsIncompleteError(
            f"system {result.system_id!r} has status {result.status!r}"
        )
    if result.row_count != expected_row_count or total != expected_row_count or errored != 0:
        raise FoundryOpenAIEvalsIncompleteError(
            f"system {result.system_id!r} expected {expected_row_count} completed rows; "
            f"output_items={result.row_count}, total={total}, errored={errored}"
        )
    missing_scores = expected_score_names - set(result.score_averages)
    if missing_scores:
        raise FoundryOpenAIEvalsIncompleteError(
            f"system {result.system_id!r} is missing scores: {sorted(missing_scores)}"
        )
    incomplete_scores = {
        name: result.score_counts.get(name, 0)
        for name in expected_score_names
        if result.score_counts and result.score_counts.get(name, 0) != expected_row_count
    }
    if incomplete_scores:
        raise FoundryOpenAIEvalsIncompleteError(
            f"system {result.system_id!r} has incomplete grader coverage: "
            f"{incomplete_scores}"
        )


def _validate_openai_evals_output_items(
    output_items: list[Any],
    *,
    expected_score_names: set[str],
) -> None:
    for row_number, item in enumerate(output_items, start=1):
        scores = {
            str(_get(result, "name")): _get(result, "score")
            for result in _get(item, "results", [])
        }
        missing = expected_score_names - set(scores)
        invalid = {
            name
            for name in expected_score_names & set(scores)
            if not isinstance(scores[name], int | float)
        }
        if missing or invalid:
            raise FoundryOpenAIEvalsIncompleteError(
                f"Foundry output row {row_number} has incomplete grader results; "
                f"missing={sorted(missing)}, invalid={sorted(invalid)}"
            )


def _validate_openai_evals_batch(
    results: list[FoundryOpenAIEvalsResult],
    *,
    config: ExperimentConfig,
    systems: list[str],
    dataset_path: Path,
) -> None:
    if [result.system_id for result in results] != systems:
        raise FoundryOpenAIEvalsIncompleteError(
            "Foundry publish did not produce exactly one result for each configured system"
        )
    for result in results:
        _, expected_row_count = _openai_evals_dataset_content(
            dataset_path,
            system_id=result.system_id,
        )
        _validate_completed_openai_evals_result(
            result,
            expected_row_count=expected_row_count,
            expected_score_names=_testing_criteria_names(config.evaluation.foundry),
        )


def _openai_evals_publish_signature(
    *,
    config: ExperimentConfig,
    evaluation_name: str,
    run_name: str,
    system_id: str,
    run_dataset_path: Path,
) -> str:
    payload = {
        "dataset_sha256": hashlib.sha256(run_dataset_path.read_bytes()).hexdigest(),
        "evaluation_name": evaluation_name,
        "run_name": run_name,
        "system_id": system_id,
        "testing_criteria": _testing_criteria(config.evaluation.foundry.managed_evaluators),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"foundry-publish-{digest[:16]}"


def _testing_criteria_names(config: FoundryEvaluationConfig) -> set[str]:
    return {str(criterion["name"]) for criterion in _testing_criteria(config.managed_evaluators)}


def _jsonl_row_count(path: Path) -> int:
    with path.open(encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def _is_transient_openai_evals_error(exc: Exception) -> bool:
    message = f"{exc.__class__.__name__}: {exc}".lower()
    return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)


def _is_missing_remote_run_error(exc: Exception) -> bool:
    message = f"{exc.__class__.__name__}: {exc}".lower()
    return "404" in message or "not found" in message


def _openai_eval_run_requires_replacement(run: Any) -> bool:
    status = str(_get(run, "status", "")).lower()
    counts = _dump_model(_get(run, "result_counts", {}))
    return status in {"failed", "canceled"} or int(counts.get("errored", 0)) > 0


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
        elif key == "document_retrieval":
            criteria.extend(
                [
                    _python_criterion(
                        name="document_retrieval_precision",
                        source=DOCUMENT_RETRIEVAL_PRECISION_GRADER_SOURCE,
                    ),
                    _python_criterion(
                        name="document_retrieval_recall",
                        source=DOCUMENT_RETRIEVAL_RECALL_GRADER_SOURCE,
                    ),
                ]
            )
        elif key == "retrieval":
            criteria.extend(
                [
                    _python_criterion(
                        name="context_precision_at_4",
                        source=CONTEXT_PRECISION_AT_4_GRADER_SOURCE,
                    ),
                    _python_criterion(
                        name="context_recall_at_4",
                        source=CONTEXT_RECALL_AT_4_GRADER_SOURCE,
                    ),
                    _python_criterion(
                        name="evidence_quote_recall_at_4",
                        source=EVIDENCE_QUOTE_RECALL_AT_4_GRADER_SOURCE,
                    ),
                ]
            )
        elif key == "groundedness":
            criteria.append(
                _python_criterion(
                    name="citation_support_rate",
                    source=CITATION_SUPPORT_RATE_GRADER_SOURCE,
                    pass_threshold=0.8,
                )
            )
        elif key == "response_completeness":
            criteria.append(
                _python_criterion(
                    name="answer_reference_token_recall",
                    source=ANSWER_REFERENCE_TOKEN_RECALL_GRADER_SOURCE,
                )
            )
        elif key == "relevance":
            criteria.append(
                _python_criterion(
                    name="answer_reference_token_f1",
                    source=ANSWER_REFERENCE_TOKEN_F1_GRADER_SOURCE,
                )
            )
        else:
            raise ValueError(
                f"unsupported portal-visible Foundry evaluator {name!r}; "
                "supported values are f1, rouge, bleu, gleu, meteor, groundedness, "
                "relevance, retrieval, document_retrieval, and response_completeness"
            )
    return criteria


def _python_criterion(
    *,
    name: str,
    source: str,
    pass_threshold: float = 0.5,
) -> dict[str, Any]:
    return {
        "type": "python",
        "name": name,
        "source": source,
        "image_tag": "2025-05-08",
        "pass_threshold": pass_threshold,
    }


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
    system_id: str,
) -> FoundryOpenAIEvalsResult:
    averages, ranges, counts = _score_summary(output_items)
    return FoundryOpenAIEvalsResult(
        evaluation_name=evaluation_name,
        eval_id=_get_required(eval_obj, "id"),
        run_name=run_name,
        run_id=_get_required(run, "id"),
        system_id=system_id,
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
        score_counts=counts,
        row_count=len(output_items),
        metadata={
            "experiment_id": config.experiment.id,
            "evaluator_path": "openai_evals",
            "system_id": system_id,
        },
    )


def _score_summary(
    output_items: list[Any],
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, int]]:
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
    counts = {name: len(values) for name, values in sorted(values_by_name.items())}
    return averages, ranges, counts


def _openai_evaluation_name(config: ExperimentConfig, configured: str | None) -> str:
    base = configured or config.evaluation.foundry.evaluation_name or config.experiment.id
    return base


def _openai_system_run_name(
    config: ExperimentConfig,
    configured: str | None,
    system_id: str,
) -> str:
    if configured is not None and "{system_id}" in configured:
        return configured.format(system_id=system_id)
    base = configured or f"{config.experiment.id}-precomputed-run"
    return f"{base}-{system_id}"


def _evaluation_metadata(
    config: ExperimentConfig,
    foundry_config: FoundryEvaluationConfig,
    *,
    system_id: str | None = None,
) -> dict[str, str]:
    metadata = {
        "experiment_id": config.experiment.id,
        "dataset_adapter": config.dataset.adapter,
        "systems": system_id or ",".join(config.systems),
        "trigger_type": "oneoff",
    }
    if system_id is not None:
        metadata["system_id"] = system_id
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
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return dict(attributes)
    return {}
