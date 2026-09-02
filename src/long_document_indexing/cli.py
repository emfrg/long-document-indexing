from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import Annotated

import typer

from long_document_indexing.config import ExperimentConfig, load_experiment_config
from long_document_indexing.datasets.base import LoadedDataset
from long_document_indexing.datasets.registry import create_dataset_adapter
from long_document_indexing.domain.maps import IndexArtifact
from long_document_indexing.domain.runs import MetricRecord, RagRunRecord
from long_document_indexing.evaluation.local.maps import evaluate_index_artifact
from long_document_indexing.evaluation.runner import aggregate_metric_means, evaluate_run
from long_document_indexing.models.fake import FakeTextGenerationClient
from long_document_indexing.prompts import PromptLoader
from long_document_indexing.retrieval.local_vector import LocalVectorBackend
from long_document_indexing.services import Services
from long_document_indexing.storage.artifacts import ArtifactStore
from long_document_indexing.systems.registry import create_system
from long_document_indexing.telemetry.usage import UsageEvent, UsageLedger
from long_document_indexing.workflows.common_indexing import run_indexing_workflow
from long_document_indexing.workflows.common_query import run_query_workflow
from long_document_indexing.workflows.execution import LocalWorkflowRunner

app = typer.Typer(no_args_is_help=True)
ConfigPath = Annotated[Path, typer.Option("--config", "-c")]


@app.command()
def prepare(config: ConfigPath) -> None:
    """Validate config and dataset, then write an experiment manifest."""

    _prepare(load_experiment_config(config))


@app.command()
def index(config: ConfigPath) -> None:
    """Build index artifacts for all configured systems and corpora."""

    asyncio.run(_index(load_experiment_config(config)))


@app.command()
def query(config: ConfigPath) -> None:
    """Run benchmark questions through existing index artifacts."""

    asyncio.run(_query(load_experiment_config(config)))


@app.command()
def evaluate(config: ConfigPath) -> None:
    """Calculate deterministic local metrics from run records."""

    _evaluate(load_experiment_config(config))


@app.command()
def report(config: ConfigPath) -> None:
    """Aggregate local metrics into CSV and Markdown summaries."""

    _report(load_experiment_config(config))


@app.command()
def run(config: ConfigPath) -> None:
    """Execute prepare, index, query, evaluate, and report."""

    experiment_config = load_experiment_config(config)
    _prepare(experiment_config)
    asyncio.run(_index(experiment_config))
    asyncio.run(_query(experiment_config))
    _evaluate(experiment_config)
    _report(experiment_config)


def _prepare(config: ExperimentConfig) -> None:
    loaded = _load_dataset(config)
    _validate_loaded_dataset(loaded)
    store = _artifact_store(config)
    store.write_json(
        "manifest.json",
        {
            "experiment": config.experiment.model_dump(mode="json"),
            "dataset": config.dataset.model_dump(mode="json"),
            "systems": config.systems,
            "corpus_ids": [corpus.id for corpus in loaded.corpora],
            "question_count": len(loaded.question_set.items),
        },
    )
    typer.echo(f"Prepared experiment {config.experiment.id}")


async def _index(config: ExperimentConfig) -> None:
    loaded = _load_dataset(config)
    services = _services(config)
    artifacts: list[IndexArtifact] = []

    for system_id in config.systems:
        system = create_system(system_id)
        system_artifacts: list[IndexArtifact] = []
        for corpus in loaded.corpora:
            artifact = await run_indexing_workflow(
                system=system,
                corpus=corpus,
                services=services,
                pipeline=config.shared_pipeline,
                experiment_id=config.experiment.id,
            )
            artifacts.append(artifact)
            system_artifacts.append(artifact)
        services.artifact_store.write_jsonl(
            f"indexes/{system.id}/index_artifacts.jsonl",
            system_artifacts,
        )

    services.artifact_store.write_jsonl("indexes/index_artifacts.jsonl", artifacts)
    _write_usage(services, "costs/index-usage.jsonl")
    typer.echo(f"Indexed {len(artifacts)} system/corpus pair(s)")


async def _query(config: ExperimentConfig) -> None:
    loaded = _load_dataset(config)
    services = _services(config)
    artifacts = _load_index_artifacts(services.artifact_store)
    corpora_by_id = {corpus.id: corpus for corpus in loaded.corpora}

    for system_id in config.systems:
        system = create_system(system_id)
        records: list[RagRunRecord] = []
        for artifact in artifacts:
            if artifact.system_id != system.id:
                continue
            corpus = corpora_by_id[artifact.corpus_id]
            items = [
                item for item in loaded.question_set.items if item.corpus_id == artifact.corpus_id
            ]
            for item in items:
                for repetition in range(config.experiment.query_repetitions):
                    records.append(
                        await run_query_workflow(
                            system=system,
                            item=item,
                            corpus=corpus,
                            index_artifact=artifact,
                            services=services,
                            pipeline=config.shared_pipeline,
                            experiment_id=config.experiment.id,
                            repetition=repetition,
                        )
                    )
        services.artifact_store.write_jsonl(f"runs/{system.id}.jsonl", records)
        typer.echo(f"Queried {len(records)} item run(s) for {system.id}")
    _write_usage(services, "costs/query-usage.jsonl")


def _evaluate(config: ExperimentConfig) -> None:
    loaded = _load_dataset(config)
    store = _artifact_store(config)
    corpora_by_id = {corpus.id: corpus for corpus in loaded.corpora}
    items_by_id = {item.id: item for item in loaded.question_set.items}

    metrics: list[MetricRecord] = []
    artifacts = _load_index_artifacts(store)
    for system_id in config.systems:
        system = create_system(system_id)
        records = [
            RagRunRecord.model_validate(row) for row in store.read_jsonl(f"runs/{system.id}.jsonl")
        ]
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


def _report(config: ExperimentConfig) -> None:
    store = _artifact_store(config)
    metrics = [
        MetricRecord.model_validate(row)
        for row in store.read_jsonl("evaluations/local-metrics.jsonl")
    ]
    rows = aggregate_metric_means(metrics)

    csv_path = store.path("report/results.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system_id", "metric", "mean", "count"])
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = store.path("report/results.md")
    markdown_path.write_text(_markdown_table(rows), encoding="utf-8")
    typer.echo(f"Wrote report to {markdown_path}")


def _load_dataset(config: ExperimentConfig) -> LoadedDataset:
    adapter = create_dataset_adapter(config.dataset)
    return adapter.load()


def _artifact_store(config: ExperimentConfig) -> ArtifactStore:
    return ArtifactStore(config.storage.artifacts_dir, config.experiment.id)


def _services(config: ExperimentConfig) -> Services:
    store = _artifact_store(config)
    retrieval_backend = LocalVectorBackend(store.path("indexes", "local_vector"))
    return Services(
        artifact_store=store,
        retrieval_backend=retrieval_backend,
        workflow_runner=LocalWorkflowRunner(),
        usage_ledger=UsageLedger(),
        prompt_loader=PromptLoader(Path("prompts")),
        generator_client=FakeTextGenerationClient(),
    )


def _load_index_artifacts(store: ArtifactStore) -> list[IndexArtifact]:
    rows = store.read_jsonl("indexes/index_artifacts.jsonl")
    if not rows:
        raise FileNotFoundError("no index artifacts found; run `ldi index` first")
    return [IndexArtifact.model_validate(row) for row in rows]


def _validate_loaded_dataset(loaded: LoadedDataset) -> None:
    corpus_ids = {corpus.id for corpus in loaded.corpora}
    missing = [item.id for item in loaded.question_set.items if item.corpus_id not in corpus_ids]
    if missing:
        raise ValueError(f"question items reference unknown corpora: {missing}")


def _write_usage(services: Services, relative_path: str) -> None:
    records: list[UsageEvent] = services.usage_ledger.records
    services.artifact_store.write_jsonl(relative_path, records)


def _markdown_table(rows: list[dict[str, str | float]]) -> str:
    lines = [
        "# Local Metric Summary",
        "",
        "| system_id | metric | mean | count |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['system_id']} | {row['metric']} | {row['mean']:.4f} | {int(row['count'])} |"
        )
    lines.append("")
    return "\n".join(lines)
