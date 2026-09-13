from __future__ import annotations

import asyncio
import csv
import json
from collections.abc import Coroutine
from pathlib import Path
from typing import Annotated, Any

import typer

from long_document_indexing.config import (
    ExperimentConfig,
    RunControlConfig,
    load_experiment_config,
)
from long_document_indexing.datasets.base import LoadedDataset
from long_document_indexing.datasets.registry import create_dataset_adapter
from long_document_indexing.domain.maps import IndexArtifact
from long_document_indexing.domain.runs import MetricRecord, RagRunRecord, UsageRecord
from long_document_indexing.evaluation.foundry import (
    FoundryEvaluationExport,
    build_foundry_managed_evaluation_plan,
    build_foundry_openai_evals_plan,
    run_foundry_managed_evaluation,
    run_foundry_openai_evals_per_system,
    write_foundry_evaluation_export,
)
from long_document_indexing.evaluation.local.maps import evaluate_index_artifact
from long_document_indexing.evaluation.runner import evaluate_run
from long_document_indexing.models.factory import (
    create_embedding_client,
    create_text_generation_client,
    generation_deployment_for_role,
)
from long_document_indexing.prompt_safety import PROMPT_SAFETY_POLICY_VERSION
from long_document_indexing.prompts import PromptLoader
from long_document_indexing.reporting import (
    build_report_bundle,
    legacy_metric_csv_rows,
    render_markdown_report,
    system_summary_csv_rows,
    usage_summary_csv_rows,
)
from long_document_indexing.retrieval.dense_vector import DenseVectorBackend
from long_document_indexing.retrieval.local_vector import LocalVectorBackend
from long_document_indexing.routing import ROUTING_POLICY_VERSION
from long_document_indexing.run_control import (
    BudgetExceeded,
    BudgetLedger,
    describe_budget,
)
from long_document_indexing.services import Services
from long_document_indexing.storage.artifacts import ArtifactStore
from long_document_indexing.storage.maps import read_document_map
from long_document_indexing.systems.map_base import (
    MAP_SOURCE_REFERENCE_NORMALIZATION_POLICY_VERSION,
)
from long_document_indexing.systems.registry import create_system
from long_document_indexing.telemetry.tracing import stable_id, stable_query_run_id
from long_document_indexing.telemetry.usage import UsageEvent, UsageLedger
from long_document_indexing.workflows.common_indexing import run_indexing_workflow
from long_document_indexing.workflows.common_query import (
    QUERY_RUN_POLICY_VERSION,
    index_artifact_signature,
    run_query_workflow,
)
from long_document_indexing.workflows.execution import LocalWorkflowRunner, MafWorkflowRunner

app = typer.Typer(no_args_is_help=True)
ConfigPath = Annotated[Path, typer.Option("--config", "-c")]
ResumeFlag = Annotated[
    bool,
    typer.Option("--resume", help="Reuse succeeded index artifacts and query records."),
]
ForceFlag = Annotated[
    bool,
    typer.Option("--force", help="Ignore reusable artifacts and rebuild or requery work."),
]
DryRunBudgetFlag = Annotated[
    bool,
    typer.Option("--dry-run-budget", help="Print run-control budget status without executing."),
]
DryRunFlag = Annotated[
    bool,
    typer.Option("--dry-run", help="Print the planned operation without executing it."),
]
EvaluationNameOption = Annotated[
    str | None,
    typer.Option("--evaluation-name", help="Foundry Evals parent name to create or reuse."),
]
RunNameOption = Annotated[
    str | None,
    typer.Option("--run-name", help="Foundry Evals run name to create."),
]

@app.command()
def prepare(config: ConfigPath) -> None:
    """Validate config and dataset, then write an experiment manifest."""

    _prepare(load_experiment_config(config))


@app.command()
def index(
    config: ConfigPath,
    resume: ResumeFlag = False,
    force: ForceFlag = False,
    dry_run_budget: DryRunBudgetFlag = False,
) -> None:
    """Build index artifacts for all configured systems and corpora."""

    experiment_config = load_experiment_config(config)
    run_control = _effective_run_control(experiment_config, resume=resume, force=force)
    experiment_config = experiment_config.model_copy(update={"run_control": run_control})
    if dry_run_budget:
        _dry_run_budget(experiment_config, run_control)
        return
    _run_with_budget_handling(_index(experiment_config, run_control=run_control))


@app.command()
def query(
    config: ConfigPath,
    resume: ResumeFlag = False,
    force: ForceFlag = False,
    dry_run_budget: DryRunBudgetFlag = False,
) -> None:
    """Run benchmark questions through existing index artifacts."""

    experiment_config = load_experiment_config(config)
    run_control = _effective_run_control(experiment_config, resume=resume, force=force)
    experiment_config = experiment_config.model_copy(update={"run_control": run_control})
    if dry_run_budget:
        _dry_run_budget(experiment_config, run_control)
        return
    _run_with_budget_handling(_query(experiment_config, run_control=run_control))


@app.command()
def evaluate(config: ConfigPath) -> None:
    """Calculate deterministic local metrics from run records."""

    _evaluate(load_experiment_config(config))


@app.command("export-foundry-eval")
def export_foundry_eval(config: ConfigPath) -> None:
    """Export run records as a Foundry-ready JSONL evaluation dataset."""

    _export_foundry_eval(load_experiment_config(config))


@app.command("evaluate-foundry-managed")
def evaluate_foundry_managed(config: ConfigPath, dry_run: DryRunFlag = False) -> None:
    """Run Azure AI Evaluation SDK over the exported Foundry dataset."""

    _evaluate_foundry_managed(load_experiment_config(config), dry_run=dry_run)


@app.command("publish-foundry-evals")
def publish_foundry_evals(
    config: ConfigPath,
    dry_run: DryRunFlag = False,
    evaluation_name: EvaluationNameOption = None,
    run_name: RunNameOption = None,
) -> None:
    """Create portal-visible Foundry Evals runs, one per benchmark system."""

    _publish_foundry_evals(
        load_experiment_config(config),
        dry_run=dry_run,
        evaluation_name=evaluation_name,
        run_name=run_name,
    )


@app.command()
def report(config: ConfigPath) -> None:
    """Aggregate local metrics into CSV and Markdown summaries."""

    _report(load_experiment_config(config))


@app.command()
def run(
    config: ConfigPath,
    resume: ResumeFlag = False,
    force: ForceFlag = False,
    dry_run_budget: DryRunBudgetFlag = False,
) -> None:
    """Execute prepare, index, query, evaluate, and report."""

    experiment_config = load_experiment_config(config)
    run_control = _effective_run_control(experiment_config, resume=resume, force=force)
    experiment_config = experiment_config.model_copy(update={"run_control": run_control})
    if dry_run_budget:
        _dry_run_budget(experiment_config, run_control)
        return

    _prepare(experiment_config)
    budget = _budget_ledger(
        _artifact_store(experiment_config),
        run_control,
        include_existing_usage=run_control.resume,
    )
    _run_with_budget_handling(
        _run_index_and_query(experiment_config, run_control=run_control, budget=budget)
    )
    _evaluate(experiment_config)
    _report(experiment_config)


async def _run_index_and_query(
    config: ExperimentConfig,
    *,
    run_control: RunControlConfig,
    budget: BudgetLedger,
) -> None:
    await _index(config, run_control=run_control, budget=budget)
    await _query(config, run_control=run_control, budget=budget)


def _run_with_budget_handling(coro: Coroutine[Any, Any, None]) -> None:
    try:
        asyncio.run(coro)
    except BudgetExceeded as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


def _effective_run_control(
    config: ExperimentConfig,
    *,
    resume: bool,
    force: bool,
) -> RunControlConfig:
    if resume and force:
        raise typer.BadParameter("--resume and --force cannot be used together")

    payload = config.run_control.model_dump()
    if resume:
        payload["resume"] = True
        payload["force"] = False
    if force:
        payload["resume"] = False
        payload["force"] = True
    return RunControlConfig.model_validate(payload)


def _dry_run_budget(config: ExperimentConfig, run_control: RunControlConfig) -> None:
    loaded = _load_dataset(config)
    services = _services(config)
    experiment_dir = config.storage.artifacts_dir / config.experiment.id
    existing_index_events = _load_usage_events_from_dir(experiment_dir, "costs/index-usage.jsonl")
    existing_query_events = _load_usage_events_from_dir(experiment_dir, "costs/query-usage.jsonl")
    initial_events = [*existing_index_events, *existing_query_events] if run_control.resume else []
    budget = BudgetLedger(run_control, initial_events=initial_events)
    existing_artifacts = _load_index_artifacts_from_dir(experiment_dir)
    reusable_artifacts = [
        artifact
        for artifact in existing_artifacts
        if _is_reusable_index_artifact(
            artifact,
            expected_retrieval_backend=services.retrieval_backend.__class__.__name__,
            expected_index_config_signature=services.index_config_signature(artifact.system_id),
        )
    ]
    existing_runs = _load_run_records_from_dir(experiment_dir, config)
    successful_runs = [record for record in existing_runs if record.status == "succeeded"]
    artifacts_by_key = {_index_artifact_key(artifact): artifact for artifact in reusable_artifacts}
    reusable_runs = [
        record
        for record in successful_runs
        if (
            artifact := artifacts_by_key.get((record.system_id, record.corpus_id))
        ) is not None
        and _is_reusable_query_record(
            record,
            artifact,
            expected_query_config_signature=services.query_config_signature,
        )
    ]
    planned_index_units = len(config.systems) * len(loaded.corpora)
    planned_query_units = (
        len(config.systems) * len(loaded.question_set.items) * config.experiment.query_repetitions
    )

    typer.echo(f"Experiment: {config.experiment.id}")
    typer.echo(f"Run control: resume={run_control.resume}, force={run_control.force}")
    typer.echo(
        "Planned units: "
        f"index={planned_index_units}, query={planned_query_units}, "
        f"existing_indexes={len(existing_artifacts)}, "
        f"reusable_indexes={len(reusable_artifacts)}, "
        f"existing_successful_queries={len(successful_runs)}, "
        f"reusable_queries={len(reusable_runs)}"
    )
    for line in describe_budget(run_control, budget.snapshot):
        typer.echo(line)


def _prepare(config: ExperimentConfig) -> None:
    loaded = _load_dataset(config)
    _validate_loaded_dataset(loaded)
    store = _artifact_store(config)
    store.write_json(
        "manifest.json",
        {
            "experiment": config.experiment.model_dump(mode="json"),
            "dataset": config.dataset.model_dump(mode="json"),
            "models": config.models.model_dump(mode="json"),
            "model_roles": _resolved_model_roles(config),
            "systems": config.systems,
            "system_configs": {
                system_id: config.system_config_for(system_id).model_dump(mode="json")
                for system_id in config.systems
            },
            "answering": config.answering.model_dump(mode="json"),
            "shared_pipeline": config.shared_pipeline.model_dump(mode="json"),
            "workflow": config.workflow.model_dump(mode="json"),
            "evaluation": config.evaluation.model_dump(mode="json"),
            "run_control": config.run_control.model_dump(mode="json"),
            "corpus_ids": [corpus.id for corpus in loaded.corpora],
            "question_count": len(loaded.question_set.items),
        },
    )
    typer.echo(f"Prepared experiment {config.experiment.id}")


async def _index(
    config: ExperimentConfig,
    *,
    run_control: RunControlConfig | None = None,
    budget: BudgetLedger | None = None,
) -> None:
    loaded = _load_dataset(config)
    services = _services(config)
    run_control = run_control or config.run_control
    budget = budget or _budget_ledger(
        services.artifact_store,
        run_control,
        include_existing_usage=run_control.resume,
    )
    existing_artifacts = (
        _load_index_artifacts_if_present(services.artifact_store)
        if run_control.resume and not run_control.force
        else []
    )
    artifacts_by_key = {
        _index_artifact_key(artifact): artifact
        for artifact in existing_artifacts
        if _is_reusable_index_artifact(
            artifact,
            expected_retrieval_backend=services.retrieval_backend.__class__.__name__,
            expected_index_config_signature=services.index_config_signature(artifact.system_id),
        )
    }
    indexed = 0
    reused = 0

    for system_id in config.systems:
        system = _create_system(config, system_id)
        for corpus in loaded.corpora:
            key = (system.id, corpus.id)
            if key in artifacts_by_key:
                reused += 1
                continue

            label = f"index {system.id}/{corpus.id}"
            budget.require_available(label)
            artifact = await run_indexing_workflow(
                system=system,
                corpus=corpus,
                services=services,
                pipeline=config.shared_pipeline,
                experiment_id=config.experiment.id,
            )
            artifacts_by_key[key] = artifact
            indexed += 1
            budget.add_usage(_usage_from_index_artifact(artifact))
            _write_index_artifacts(config, loaded, services.artifact_store, artifacts_by_key)
            _write_usage(
                services,
                "costs/index-usage.jsonl",
                preserve_existing=run_control.resume and not run_control.force,
            )
            budget.require_not_exceeded(label)

    _write_index_artifacts(config, loaded, services.artifact_store, artifacts_by_key)
    _write_usage(
        services,
        "costs/index-usage.jsonl",
        preserve_existing=run_control.resume and not run_control.force,
    )
    message = f"Indexed {indexed} system/corpus pair(s)"
    if reused:
        message += f"; reused {reused}"
    typer.echo(message)


async def _query(
    config: ExperimentConfig,
    *,
    run_control: RunControlConfig | None = None,
    budget: BudgetLedger | None = None,
) -> None:
    loaded = _load_dataset(config)
    services = _services(config)
    run_control = run_control or config.run_control
    budget = budget or _budget_ledger(
        services.artifact_store,
        run_control,
        include_existing_usage=run_control.resume,
    )
    artifacts = _load_index_artifacts(services.artifact_store)
    corpora_by_id = {corpus.id: corpus for corpus in loaded.corpora}

    for system_id in config.systems:
        system = _create_system(config, system_id)
        records_by_run_id: dict[str, RagRunRecord] = (
            {
                record.run_id: record
                for record in _load_system_run_records(services.artifact_store, system.id)
            }
            if run_control.resume and not run_control.force
            else {}
        )
        queried = 0
        reused = 0
        skipped = 0
        for artifact in artifacts:
            if artifact.system_id != system.id:
                continue
            corpus = corpora_by_id[artifact.corpus_id]
            items = [
                item for item in loaded.question_set.items if item.corpus_id == artifact.corpus_id
            ]
            for item in items:
                for repetition in range(config.experiment.query_repetitions):
                    run_id = stable_query_run_id(
                        config.experiment.id,
                        system.id,
                        item.id,
                        repetition,
                    )
                    existing = records_by_run_id.get(run_id)
                    if existing is not None and _is_reusable_query_record(
                        existing,
                        artifact,
                        expected_query_config_signature=services.query_config_signature,
                    ):
                        reused += 1
                        continue

                    label = f"query {system.id}/{item.id}/rep-{repetition}"
                    exhausted_reason = budget.exhausted_reason(label)
                    if exhausted_reason is not None:
                        records_by_run_id[run_id] = _skipped_run_record(
                            config,
                            system_id=system.id,
                            corpus_id=corpus.id,
                            item_id=item.id,
                            repetition=repetition,
                            index_artifact=artifact,
                            query_config_signature=services.query_config_signature,
                            reason=exhausted_reason,
                        )
                        skipped += 1
                        _write_system_run_records(
                            services.artifact_store,
                            system.id,
                            records_by_run_id,
                        )
                        continue

                    record = await run_query_workflow(
                        system=system,
                        item=item,
                        corpus=corpus,
                        index_artifact=artifact,
                        services=services,
                        pipeline=config.shared_pipeline,
                        experiment_id=config.experiment.id,
                        repetition=repetition,
                    )
                    records_by_run_id[run_id] = record
                    queried += 1
                    budget.add_usage(record.usage)
                    _write_system_run_records(
                        services.artifact_store,
                        system.id,
                        records_by_run_id,
                    )
                    _write_usage(
                        services,
                        "costs/query-usage.jsonl",
                        preserve_existing=run_control.resume and not run_control.force,
                    )
                    budget.require_not_exceeded(label)
        _write_system_run_records(services.artifact_store, system.id, records_by_run_id)
        message = f"Queried {queried} item run(s) for {system.id}"
        if reused:
            message += f"; reused {reused}"
        if skipped:
            message += f"; skipped {skipped}"
        typer.echo(message)
    _write_usage(
        services,
        "costs/query-usage.jsonl",
        preserve_existing=run_control.resume and not run_control.force,
    )


def _evaluate(config: ExperimentConfig) -> None:
    loaded = _load_dataset(config)
    store = _artifact_store(config)
    corpora_by_id = {corpus.id: corpus for corpus in loaded.corpora}
    items_by_id = {item.id: item for item in loaded.question_set.items}

    metrics: list[MetricRecord] = []
    foundry_records: list[RagRunRecord] = []
    artifacts = _load_index_artifacts(store)
    for system_id in config.systems:
        system = _create_system(config, system_id)
        records = [
            RagRunRecord.model_validate(row) for row in store.read_jsonl(f"runs/{system.id}.jsonl")
        ]
        foundry_records.extend(records)
        for record in records:
            metrics.extend(
                evaluate_run(
                    record,
                    items_by_id[record.item_id],
                    corpora_by_id[record.corpus_id],
                    config.evaluation.local,
                )
            )
        for artifact in artifacts:
            if artifact.system_id != system.id:
                continue
            metrics.extend(
                evaluate_index_artifact(
                    artifact,
                    corpora_by_id[artifact.corpus_id],
                    store,
                    config.evaluation.local,
                    experiment_id=config.experiment.id,
                )
            )

    store.write_jsonl("evaluations/local-metrics.jsonl", metrics)
    typer.echo(f"Evaluated {len(metrics)} local metric record(s)")
    if config.evaluation.foundry.enabled:
        _export_foundry_eval(
            config,
            loaded=loaded,
            store=store,
            records=foundry_records,
        )


def _report(config: ExperimentConfig) -> None:
    store = _artifact_store(config)
    metrics = [
        MetricRecord.model_validate(row)
        for row in store.read_jsonl("evaluations/local-metrics.jsonl")
    ]
    bundle = build_report_bundle(
        metrics=metrics,
        run_records=_load_run_records_for_report(config, store),
        index_usage=_load_usage_events(store, "costs/index-usage.jsonl"),
        query_usage=_load_usage_events(store, "costs/query-usage.jsonl"),
        foundry_manifest=_load_foundry_manifest(config, store),
        foundry_manifest_path=_foundry_manifest_path(config, store),
        foundry_managed_result=_load_foundry_managed_result(config, store),
        foundry_managed_result_path=_foundry_managed_result_path(config, store),
    )

    _write_csv(
        store.path("report/results.csv"),
        ["system_id", "metric", "mean", "count"],
        legacy_metric_csv_rows(bundle.metric_rows),
    )
    _write_csv(
        store.path("report/system-summary.csv"),
        [
            "system_id",
            "query_runs",
            "failed_runs",
            "skipped_runs",
            "quality_score",
            "routing_score",
            "retrieval_score",
            "answer_score",
            "map_score",
            "query_duration_ms_mean",
            "tool_calls_mean",
            "index_model_calls",
            "query_model_calls",
            "total_model_calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "estimated_cost",
            "issues",
        ],
        system_summary_csv_rows(bundle.system_rows),
    )
    _write_csv(
        store.path("report/usage-summary.csv"),
        [
            "system_id",
            "phase",
            "kind",
            "records",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "model_calls",
            "tool_calls",
            "duration_ms",
            "estimated_cost",
        ],
        usage_summary_csv_rows(bundle.usage_rows),
    )
    store.write_json("report/summary.json", bundle)

    markdown_path = store.path("report/results.md")
    markdown_path.write_text(render_markdown_report(bundle), encoding="utf-8")
    typer.echo(f"Wrote report to {markdown_path}")


def _export_foundry_eval(
    config: ExperimentConfig,
    *,
    loaded: LoadedDataset | None = None,
    store: ArtifactStore | None = None,
    records: list[RagRunRecord] | None = None,
) -> FoundryEvaluationExport:
    loaded = loaded or _load_dataset(config)
    store = store or _artifact_store(config)
    records = records if records is not None else _load_run_records(config, store)
    items_by_id = {item.id: item for item in loaded.question_set.items}
    export = write_foundry_evaluation_export(
        store=store,
        records=records,
        items_by_id=items_by_id,
        config=config.evaluation.foundry,
        experiment_id=config.experiment.id,
    )
    typer.echo(
        f"Wrote Foundry evaluation dataset with {export.row_count} row(s) to {export.dataset_path}"
    )
    typer.echo(f"Wrote Foundry evaluation manifest to {export.manifest_path}")
    return export


def _evaluate_foundry_managed(
    config: ExperimentConfig,
    *,
    dry_run: bool = False,
) -> None:
    store = _artifact_store(config)
    if dry_run:
        plan = build_foundry_managed_evaluation_plan(config=config, store=store)
        store.write_json("evaluations/foundry/managed-plan.json", plan)
        typer.echo(f"Planned Foundry managed evaluation for {plan['evaluation_name']}")
        typer.echo(f"Dataset: {plan['dataset_path']}")
        typer.echo(f"Manifest: {plan['manifest_path']}")
        typer.echo(f"Result: {plan['result_path']}")
        typer.echo(f"Azure AI project: {plan['azure_ai_project']}")
        typer.echo(f"Managed evaluators: {', '.join(plan['managed_evaluators'])}")
        typer.echo(f"Managed execution: {plan['managed_execution']}")
        typer.echo(
            "Managed evaluator delay seconds: "
            f"{plan['managed_evaluator_delay_seconds']}"
        )
        typer.echo(f"Managed max attempts: {plan['managed_max_attempts']}")
        typer.echo(
            "Managed retry delay seconds: "
            f"{plan['managed_retry_delay_seconds']}"
        )
        typer.echo(f"Fail on evaluator errors: {plan['fail_on_evaluator_errors']}")
        typer.echo(
            "Wrote managed evaluation plan to "
            f"{store.path('evaluations/foundry/managed-plan.json')}"
        )
        return

    _ensure_foundry_export(config, store)
    result = run_foundry_managed_evaluation(
        config=config,
        store=store,
        status_callback=lambda message: typer.echo(f"Foundry managed: {message}"),
    )
    typer.echo(f"Wrote Foundry managed evaluation result to {result.result_path}")
    if result.studio_url:
        typer.echo(f"Foundry URL: {result.studio_url}")


def _publish_foundry_evals(
    config: ExperimentConfig,
    *,
    dry_run: bool = False,
    evaluation_name: str | None = None,
    run_name: str | None = None,
) -> None:
    store = _artifact_store(config)
    _ensure_foundry_export(config, store)
    if dry_run:
        plan = build_foundry_openai_evals_plan(
            config=config,
            store=store,
            evaluation_name=evaluation_name,
            run_name=run_name,
        )
        store.write_json("evaluations/foundry/openai-evals-plan.json", plan)
        typer.echo(f"Planned Foundry Evals publish for {plan['evaluation_name']}")
        typer.echo(f"Mode: {plan['mode']}")
        typer.echo(f"Runs: {len(plan['runs'])}")
        for run in plan["runs"]:
            typer.echo(f"- {run['system_id']}: {run['run_name']}")
        typer.echo(f"Dataset: {plan['dataset_path']}")
        typer.echo(f"Result: {plan['result_path']}")
        typer.echo(f"Project endpoint: {plan['project_endpoint']}")
        typer.echo(
            "Wrote Foundry Evals publish plan to "
            f"{store.path('evaluations/foundry/openai-evals-plan.json')}"
        )
        return

    batch_result = run_foundry_openai_evals_per_system(
        config=config,
        store=store,
        evaluation_name=evaluation_name,
        run_name=run_name,
        status_callback=lambda message: typer.echo(f"Foundry Evals: {message}"),
    )
    typer.echo(f"Wrote Foundry Evals system results to {batch_result.result_path}")
    typer.echo(f"Systems: {', '.join(batch_result.systems)}")
    typer.echo(f"Rows: {batch_result.row_count}")
    for result in batch_result.results:
        typer.echo(
            f"- {result.system_id}: {result.status}, rows={result.row_count}, "
            f"run={result.run_name}"
        )
        if result.report_url:
            typer.echo(f"  Foundry report URL: {result.report_url}")


def _ensure_foundry_export(config: ExperimentConfig, store: ArtifactStore) -> None:
    dataset_path = store.experiment_dir / config.evaluation.foundry.dataset_path
    manifest_path = store.experiment_dir / config.evaluation.foundry.manifest_path
    if dataset_path.exists() and manifest_path.exists():
        return
    _export_foundry_eval(config, store=store)


def _load_dataset(config: ExperimentConfig) -> LoadedDataset:
    adapter = create_dataset_adapter(config.dataset)
    return adapter.load()


def _artifact_store(config: ExperimentConfig) -> ArtifactStore:
    return ArtifactStore(config.storage.artifacts_dir, config.experiment.id)


def _services(config: ExperimentConfig) -> Services:
    store = _artifact_store(config)
    prompt_loader = PromptLoader(Path("prompts"))
    backend_names = {
        config.system_config_for(system_id).retrieval_backend for system_id in config.systems
    }
    if len(backend_names) != 1:
        raise ValueError(
            "controlled experiments require one shared retrieval backend; "
            f"configured backends: {sorted(backend_names)}"
        )
    backend_name = next(iter(backend_names))
    embedding_client = None
    if backend_name == "dense_vector":
        embedding_client = create_embedding_client(config.models)
        retrieval_backend = DenseVectorBackend(
            store.path("indexes", "dense_vector"),
            embedding_client,
            batch_size=config.models.embedding_batch_size,
        )
    elif backend_name == "local_vector":
        retrieval_backend = LocalVectorBackend(store.path("indexes", "local_vector"))
    else:
        raise ValueError(f"unsupported retrieval backend: {backend_name}")
    generator_client = create_text_generation_client(config.models, role="generator")
    router_client = create_text_generation_client(config.models, role="router")
    answer_client = create_text_generation_client(config.models, role="answer")
    map_generation_signature = _map_generation_signature(
        config,
        prompt_loader=prompt_loader,
        generator_model_id=generator_client.model_id,
    )
    query_config_signature = _query_config_signature(
        config,
        prompt_loader=prompt_loader,
        router_model_id=router_client.model_id,
        answer_model_id=answer_client.model_id,
        embedding_model_id=embedding_client.model_id if embedding_client is not None else None,
        retrieval_backend=retrieval_backend.__class__.__name__,
    )
    index_config_signatures = {
        system_id: _index_config_signature(
            config,
            system_id=system_id,
            map_generation_signature=map_generation_signature,
            embedding_model_id=(
                embedding_client.model_id if embedding_client is not None else None
            ),
            retrieval_backend=retrieval_backend.__class__.__name__,
        )
        for system_id in config.systems
    }
    return Services(
        artifact_store=store,
        retrieval_backend=retrieval_backend,
        workflow_runner=_workflow_runner(config),
        usage_ledger=UsageLedger(),
        prompt_loader=prompt_loader,
        answering_mode=config.answering.mode,
        generator_client=generator_client,
        router_client=router_client,
        answer_client=answer_client,
        embedding_client=embedding_client,
        map_generation_signature=map_generation_signature,
        query_config_signature=query_config_signature,
        index_config_signatures=index_config_signatures,
        resume_checkpoints=config.run_control.resume and not config.run_control.force,
        progress=typer.echo,
    )


def _map_generation_signature(
    config: ExperimentConfig,
    *,
    prompt_loader: PromptLoader,
    generator_model_id: str,
) -> str:
    return _config_signature(
        "map-generation-config",
        {
            "model": _generation_role_config(config, "generator", generator_model_id),
            "prompt_safety_policy": PROMPT_SAFETY_POLICY_VERSION,
            "source_reference_normalization_policy": (
                MAP_SOURCE_REFERENCE_NORMALIZATION_POLICY_VERSION
            ),
            "prompts": _prompt_snapshot(prompt_loader, exclude={"shared/answer", "shared/route"}),
        },
    )


def _query_config_signature(
    config: ExperimentConfig,
    *,
    prompt_loader: PromptLoader,
    router_model_id: str,
    answer_model_id: str,
    embedding_model_id: str | None,
    retrieval_backend: str,
) -> str:
    return _config_signature(
        "query-config",
        {
            "router": _generation_role_config(config, "router", router_model_id),
            "answer": _generation_role_config(config, "answer", answer_model_id),
            "answering": config.answering.model_dump(mode="json"),
            "retrieval": {
                "backend": retrieval_backend,
                "embedding_model_id": embedding_model_id,
                "selected_documents": config.shared_pipeline.selected_documents,
                "retrieved_segments": config.shared_pipeline.retrieved_segments,
            },
            "routing_policy": ROUTING_POLICY_VERSION,
            "query_policy": QUERY_RUN_POLICY_VERSION,
            "prompts": {
                "shared/route": prompt_loader.load("shared", "route"),
                "shared/answer": prompt_loader.load("shared", "answer"),
            },
        },
    )


def _index_config_signature(
    config: ExperimentConfig,
    *,
    system_id: str,
    map_generation_signature: str,
    embedding_model_id: str | None,
    retrieval_backend: str,
) -> str:
    normalized_system_id = config.system_config_for(system_id).id
    return _config_signature(
        "index-config",
        {
            "system": config.system_config_for(system_id).model_dump(mode="json"),
            "retrieval": {
                "backend": retrieval_backend,
                "embedding_model_id": embedding_model_id,
                "embedding_batch_size": config.models.embedding_batch_size,
            },
            "index_pipeline": {
                "segment_tokens": config.shared_pipeline.segment_tokens,
                "segment_overlap_tokens": config.shared_pipeline.segment_overlap_tokens,
            },
            "map_generation_signature": (
                None if normalized_system_id == "flat_vector" else map_generation_signature
            ),
        },
    )


def _generation_role_config(
    config: ExperimentConfig,
    role: str,
    model_id: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "model_id": model_id,
        "provider": config.models.generator_provider,
        "api": config.models.generator_api,
        "temperature": config.models.generator_temperature,
        "max_output_tokens": config.models.generator_max_output_tokens,
        "response_format": config.models.generator_response_format,
    }


def _prompt_snapshot(prompt_loader: PromptLoader, *, exclude: set[str]) -> dict[str, str]:
    snapshot = {}
    for path in sorted(prompt_loader.root.rglob("*.md")):
        relative = path.relative_to(prompt_loader.root).with_suffix("").as_posix()
        if relative not in exclude:
            snapshot[relative] = path.read_text(encoding="utf-8")
    return snapshot


def _config_signature(prefix: str, payload: dict[str, Any]) -> str:
    return stable_id(prefix, json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _resolved_model_roles(config: ExperimentConfig) -> dict[str, str | None]:
    embedding = config.models.embedding_deployment
    if embedding is not None and (not embedding.strip() or embedding.strip().startswith("${")):
        embedding = None
    judge = config.models.judge_deployment
    if judge is not None and (not judge.strip() or judge.strip().startswith("${")):
        judge = None
    return {
        "map_builder": generation_deployment_for_role(config.models, "generator"),
        "router": generation_deployment_for_role(config.models, "router"),
        "answerer": generation_deployment_for_role(config.models, "answer"),
        "embedding": embedding,
        "judge": judge,
    }


def _workflow_runner(config: ExperimentConfig):
    if config.workflow.runner == "local":
        return LocalWorkflowRunner()
    if config.workflow.runner == "maf":
        return MafWorkflowRunner()
    raise ValueError(f"unsupported workflow runner: {config.workflow.runner}")


def _create_system(config: ExperimentConfig, system_id: str):
    return create_system(system_id, config.system_config_for(system_id))


def _load_index_artifacts(store: ArtifactStore) -> list[IndexArtifact]:
    rows = store.read_jsonl("indexes/index_artifacts.jsonl")
    if not rows:
        raise FileNotFoundError("no index artifacts found; run `ldi index` first")
    return [IndexArtifact.model_validate(row) for row in rows]


def _load_index_artifacts_if_present(store: ArtifactStore) -> list[IndexArtifact]:
    return [
        IndexArtifact.model_validate(row)
        for row in store.read_jsonl("indexes/index_artifacts.jsonl")
    ]


def _load_index_artifacts_from_dir(experiment_dir: Path) -> list[IndexArtifact]:
    path = experiment_dir / "indexes/index_artifacts.jsonl"
    if not path.exists():
        return []
    return [
        IndexArtifact.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_index_artifacts(
    config: ExperimentConfig,
    loaded: LoadedDataset,
    store: ArtifactStore,
    artifacts_by_key: dict[tuple[str, str], IndexArtifact],
) -> list[IndexArtifact]:
    ordered_artifacts = _ordered_index_artifacts(config, loaded, artifacts_by_key)
    for system_id in config.systems:
        system = _create_system(config, system_id)
        store.write_jsonl(
            f"indexes/{system.id}/index_artifacts.jsonl",
            [artifact for artifact in ordered_artifacts if artifact.system_id == system.id],
        )
    store.write_jsonl("indexes/index_artifacts.jsonl", ordered_artifacts)
    return ordered_artifacts


def _ordered_index_artifacts(
    config: ExperimentConfig,
    loaded: LoadedDataset,
    artifacts_by_key: dict[tuple[str, str], IndexArtifact],
) -> list[IndexArtifact]:
    ordered = []
    for system_id in config.systems:
        system = _create_system(config, system_id)
        for corpus in loaded.corpora:
            artifact = artifacts_by_key.get((system.id, corpus.id))
            if artifact is not None:
                ordered.append(artifact)
    return ordered


def _index_artifact_key(artifact: IndexArtifact) -> tuple[str, str]:
    return artifact.system_id, artifact.corpus_id


def _is_reusable_index_artifact(
    artifact: IndexArtifact,
    *,
    expected_retrieval_backend: str | None = None,
    expected_index_config_signature: str | None = None,
) -> bool:
    if not Path(artifact.artifact_path).exists():
        return False
    actual_backend = artifact.build_metadata.get(
        "retrieval_backend", artifact.build_metadata.get("backend")
    )
    if expected_retrieval_backend is not None and actual_backend != expected_retrieval_backend:
        return False
    if (
        expected_index_config_signature is not None
        and artifact.build_metadata.get("index_config_signature")
        != expected_index_config_signature
    ):
        return False
    paths = artifact.build_metadata.get("document_map_paths")
    if paths is None:
        return True
    if not isinstance(paths, dict):
        return False
    if not _has_current_map_artifact_policy(artifact):
        return False
    if not all(Path(str(path)).exists() for path in paths.values()):
        return False
    if expected_index_config_signature is not None:
        return _has_current_document_map_content(artifact, paths)
    return True


def _has_current_map_artifact_policy(artifact: IndexArtifact) -> bool:
    return (
        artifact.build_metadata.get("prompt_safety_policy") == PROMPT_SAFETY_POLICY_VERSION
        and artifact.build_metadata.get("source_reference_normalization_policy")
        == MAP_SOURCE_REFERENCE_NORMALIZATION_POLICY_VERSION
    )


def _has_current_document_map_content(
    artifact: IndexArtifact,
    paths: dict[str, Any],
) -> bool:
    signatures = artifact.build_metadata.get("document_map_signatures")
    if not isinstance(signatures, dict) or set(signatures) != set(paths):
        return False
    try:
        return all(
            signatures[map_id]
            == stable_id(
                "document-map-content",
                read_document_map(str(path)).model_dump_json(),
            )
            for map_id, path in paths.items()
        )
    except (OSError, ValueError):
        return False


def _is_reusable_query_record(
    record: RagRunRecord,
    artifact: IndexArtifact,
    *,
    expected_query_config_signature: str,
) -> bool:
    return (
        record.status == "succeeded"
        and record.index_artifact_id == artifact.id
        and record.index_artifact_signature == index_artifact_signature(artifact)
        and record.query_policy_version == QUERY_RUN_POLICY_VERSION
        and record.query_config_signature == expected_query_config_signature
    )


def _usage_from_index_artifact(artifact: IndexArtifact) -> UsageRecord:
    usage = artifact.build_metadata.get("usage")
    if isinstance(usage, dict):
        return UsageRecord.model_validate(usage)
    return UsageRecord()


def _load_run_records(config: ExperimentConfig, store: ArtifactStore) -> list[RagRunRecord]:
    records: list[RagRunRecord] = []
    for system_id in config.systems:
        system = _create_system(config, system_id)
        records.extend(
            RagRunRecord.model_validate(row) for row in store.read_jsonl(f"runs/{system.id}.jsonl")
        )
    if not records:
        raise FileNotFoundError("no run records found; run `ldi query` first")
    return records


def _load_system_run_records(store: ArtifactStore, system_id: str) -> list[RagRunRecord]:
    return [RagRunRecord.model_validate(row) for row in store.read_jsonl(f"runs/{system_id}.jsonl")]


def _load_run_records_from_dir(
    experiment_dir: Path,
    config: ExperimentConfig,
) -> list[RagRunRecord]:
    records: list[RagRunRecord] = []
    for system_id in config.systems:
        system = _create_system(config, system_id)
        path = experiment_dir / f"runs/{system.id}.jsonl"
        if not path.exists():
            continue
        records.extend(
            RagRunRecord.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return records


def _write_system_run_records(
    store: ArtifactStore,
    system_id: str,
    records_by_run_id: dict[str, RagRunRecord],
) -> None:
    records = sorted(
        records_by_run_id.values(),
        key=lambda record: (record.corpus_id, record.item_id, record.repetition),
    )
    store.write_jsonl(f"runs/{system_id}.jsonl", records)


def _skipped_run_record(
    config: ExperimentConfig,
    *,
    system_id: str,
    corpus_id: str,
    item_id: str,
    repetition: int,
    index_artifact: IndexArtifact,
    query_config_signature: str,
    reason: str,
) -> RagRunRecord:
    return RagRunRecord(
        run_id=stable_query_run_id(config.experiment.id, system_id, item_id, repetition),
        experiment_id=config.experiment.id,
        system_id=system_id,
        corpus_id=corpus_id,
        item_id=item_id,
        repetition=repetition,
        selected_document_ids=[],
        retrieved_items=[],
        answer="",
        citations=[],
        usage=UsageRecord(),
        index_artifact_id=index_artifact.id,
        index_artifact_signature=index_artifact_signature(index_artifact),
        query_policy_version=QUERY_RUN_POLICY_VERSION,
        query_config_signature=query_config_signature,
        status="skipped",
        error=reason,
    )


def _load_run_records_for_report(
    config: ExperimentConfig,
    store: ArtifactStore,
) -> list[RagRunRecord]:
    records: list[RagRunRecord] = []
    for system_id in config.systems:
        system = _create_system(config, system_id)
        records.extend(
            RagRunRecord.model_validate(row) for row in store.read_jsonl(f"runs/{system.id}.jsonl")
        )
    return records


def _load_usage_events(store: ArtifactStore, relative_path: str) -> list[UsageEvent]:
    return [UsageEvent.model_validate(row) for row in store.read_jsonl(relative_path)]


def _load_usage_events_from_dir(experiment_dir: Path, relative_path: str) -> list[UsageEvent]:
    path = experiment_dir / relative_path
    if not path.exists():
        return []
    return [
        UsageEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _budget_ledger(
    store: ArtifactStore,
    run_control: RunControlConfig,
    *,
    include_existing_usage: bool,
) -> BudgetLedger:
    existing_usage = (
        [
            *_load_usage_events(store, "costs/index-usage.jsonl"),
            *_load_usage_events(store, "costs/query-usage.jsonl"),
        ]
        if include_existing_usage
        else []
    )
    return BudgetLedger(run_control, initial_events=existing_usage)


def _load_foundry_manifest(
    config: ExperimentConfig,
    store: ArtifactStore,
) -> dict | None:
    manifest_relative_path = config.evaluation.foundry.manifest_path
    manifest_path = store.experiment_dir / manifest_relative_path
    if not manifest_path.exists():
        return None
    return store.read_json(manifest_relative_path)


def _foundry_manifest_path(config: ExperimentConfig, store: ArtifactStore) -> str | None:
    manifest_path = store.experiment_dir / config.evaluation.foundry.manifest_path
    if not manifest_path.exists():
        return None
    return str(manifest_path)


def _load_foundry_managed_result(
    config: ExperimentConfig,
    store: ArtifactStore,
) -> dict | None:
    result_relative_path = config.evaluation.foundry.result_path
    result_path = store.experiment_dir / result_relative_path
    if not result_path.exists():
        return None
    result = store.read_json(result_relative_path)
    if not _foundry_managed_result_matches_current_manifest(config, store, result):
        return None
    return result


def _foundry_managed_result_path(
    config: ExperimentConfig,
    store: ArtifactStore,
) -> str | None:
    result_path = store.experiment_dir / config.evaluation.foundry.result_path
    if not result_path.exists():
        return None
    result = store.read_json(config.evaluation.foundry.result_path)
    if not _foundry_managed_result_matches_current_manifest(config, store, result):
        return None
    return str(result_path)


def _foundry_managed_result_matches_current_manifest(
    config: ExperimentConfig,
    store: ArtifactStore,
    result: dict,
) -> bool:
    manifest = _load_foundry_manifest(config, store)
    if manifest is None:
        return False
    dataset_digest = manifest.get("dataset_sha256")
    result_metadata = result.get("metadata")
    if not isinstance(result_metadata, dict):
        return False
    return bool(
        dataset_digest
        and result_metadata.get("dataset_sha256") == dataset_digest
        and result_metadata.get("completion_status") == "complete"
    )


def _validate_loaded_dataset(loaded: LoadedDataset) -> None:
    corpus_ids = {corpus.id for corpus in loaded.corpora}
    missing = [item.id for item in loaded.question_set.items if item.corpus_id not in corpus_ids]
    if missing:
        raise ValueError(f"question items reference unknown corpora: {missing}")


def _write_usage(
    services: Services,
    relative_path: str,
    *,
    preserve_existing: bool = False,
) -> None:
    records: list[UsageEvent] = services.usage_ledger.records
    if preserve_existing:
        records = _merge_usage_events(
            _load_usage_events(services.artifact_store, relative_path),
            records,
        )
    services.artifact_store.write_jsonl(relative_path, records)


def _merge_usage_events(
    existing: list[UsageEvent],
    new: list[UsageEvent],
) -> list[UsageEvent]:
    merged = {_usage_event_key(event): event for event in existing}
    for event in new:
        merged[_usage_event_key(event)] = event
    return sorted(
        merged.values(),
        key=lambda event: (
            event.system_id,
            event.corpus_id,
            event.item_id or "",
            event.repetition,
            event.stage,
            event.kind,
        ),
    )


def _usage_event_key(event: UsageEvent) -> tuple[str, str, str | None, int, str, str]:
    return (
        event.run_id,
        event.stage,
        event.item_id,
        event.repetition,
        event.kind,
        event.system_id,
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
