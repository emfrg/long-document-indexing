# Repository Walkthrough

This project is a benchmark-first Python framework for comparing long-document indexing strategies for retrieval-augmented generation. It keeps the core benchmark loop independent from any one model provider, workflow framework, or evaluation backend.

## Build Order

| Milestone | Name | What It Added |
| --- | --- | --- |
| 1 | Python benchmark kernel | Pydantic domain objects, local smoke corpus, questions, retrieval baseline, run records, and deterministic metrics. |
| 2 | Workflow and Telemetry foundation | Workflow runner protocol, local workflow records, stable trace IDs, usage events, prompt loading, and service wiring. |
| 3 | First Document Maps Systems | `stuffing`, `map_reduce`, and `refine` systems that build and query document maps through the same interfaces. |
| 4 | Real Model Client thin slice | OpenAI-compatible generation client for Foundry/Azure OpenAI style endpoints. |
| 5 | MAF workflow thin slice | Optional Microsoft Agent Framework runner behind the existing workflow interface. |
| 6 | Foundry GPT-5 Structured Generation | Responses API support with Pydantic structured-output parsing for GPT-5-family deployments. |
| 7 | Model-Backed Answer Generation | Optional generated answer mode with citation validation against retrieved evidence. |
| 8 | Foundry Evaluation Export | Foundry-ready JSONL export with query, response, context, ground truth, and metadata. |
| 9 | Multi-System Real Benchmark Run | Real GPT-5-family smoke run across `flat_vector`, `stuffing`, `map_reduce`, and `refine`. |
| 10 | Analysis And Reporting Upgrade | Typed report bundle, Markdown report, CSV summaries, usage rollups, and issue summaries. |
| 11 | Larger Benchmark Dataset Thin Slice | Versioned synthetic enterprise benchmark with 8 documents, 24 source segments, and 16 questions. |
| 12 | Resumable And Budgeted Live Runs | Resume reuse, force rebuilds, dry-run budget planning, and model-call/token/cost caps. |
| 13 | Foundry Managed Evaluation Execution | Azure AI Evaluation SDK execution path over exported JSONL datasets. |
| 14 | Larger Real Benchmark Run | GPT-5-family stuffing-only enterprise run with managed `f1` and `rouge` scoring. |
| 15 | Repo Polish And Final Walkthrough | This walkthrough, README alignment, packaging metadata, and final verification. |

## Runtime Flow

1. `ldi run --config <experiment.yaml>` loads the YAML experiment through `long_document_indexing.config`.
2. The dataset adapter loads corpora and benchmark questions into stable domain models.
3. The CLI builds `Services`: artifact storage, retrieval backend, workflow runner, prompt loader, usage ledger, and optional generation client.
4. Index workflows call each configured system to create reusable `IndexArtifact` records.
5. Query workflows run benchmark questions against those artifacts and write `RagRunRecord` JSONL files.
6. Local evaluation reads the run records and writes deterministic `MetricRecord` JSONL files.
7. Optional Foundry export converts run records into a single-turn evaluation dataset.
8. Reporting aggregates metrics, run statuses, usage, Foundry export metadata, and managed-evaluation results.

## Directory Guide

`src/long_document_indexing/domain/` contains the benchmark schemas. These objects define corpora, questions, maps, run records, metric records, and usage records.

`src/long_document_indexing/datasets/` adapts benchmark assets into the domain model. The local JSON/JSONL adapter is the default path; `multilexsum` is optional for later external dataset work.

`src/long_document_indexing/systems/` contains the indexing strategies under comparison. `flat_vector` is the non-map baseline, while `stuffing`, `map_reduce`, and `refine` use document maps.

`src/long_document_indexing/workflows/` defines the execution boundary. The local runner and MAF runner both expose the same workflow result shape, so benchmark behavior does not depend on the workflow backend.

`src/long_document_indexing/models/` defines model-generation boundaries. The fake client keeps tests deterministic; the OpenAI-compatible client supports Foundry/Azure OpenAI endpoints and GPT-5-family Responses structured outputs.

`src/long_document_indexing/evaluation/` contains local metrics and Foundry adapters. Local metrics are deterministic. Foundry export and managed evaluation are opt-in.

`src/long_document_indexing/reporting.py` builds the report bundle used by `ldi report` and the end of `ldi run`.

`configs/experiments/` contains complete runnable benchmark definitions. `configs/systems/` stores system-level configuration placeholders for future richer system configs.

`benchmarks/` contains versioned benchmark fixtures. `benchmarks/smoke/` is the smallest deterministic fixture. `benchmarks/enterprise/` is the larger synthetic thin slice.

`prompts/` contains prompt templates for maps, route decisions, and generated answers.

`artifacts/` is ignored by git. It stores manifests, indexes, run JSONL, metrics, usage ledgers, Foundry exports, managed-evaluation outputs, and reports.

## Main Commands

Run the deterministic smoke benchmark:

```bash
uv run --python 3.13 --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/smoke-test.yaml
```

Run the deterministic enterprise benchmark:

```bash
uv run --python 3.13 --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/enterprise-thin-slice.yaml
```

Preview budget state before a live run:

```bash
uv run --python 3.13 --extra foundry --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/foundry-enterprise-stuffing-thin-slice.yaml --dry-run-budget
```

Run the capped real enterprise stuffing benchmark:

```bash
uv run --python 3.13 --extra foundry --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/foundry-enterprise-stuffing-thin-slice.yaml --resume
```

Run managed evaluation over an existing export:

```bash
uv run --python 3.13 --extra foundry --no-editable --reinstall-package long-document-indexing ldi evaluate-foundry-managed --config configs/experiments/enterprise-thin-slice.yaml
```

Regenerate a report from existing artifacts:

```bash
uv run --python 3.13 --no-editable --reinstall-package long-document-indexing ldi report --config configs/experiments/enterprise-thin-slice.yaml
```

## Credential Boundaries

No credentials are needed for fake-generator local runs.

Real inference needs `.env` or environment variables for:

```text
FOUNDRY_GENERATOR_BASE_URL
FOUNDRY_GENERATOR_DEPLOYMENT
AZURE_INFERENCE_CREDENTIAL
```

The base URL should end at `/openai/v1/`. If it includes `/responses` or `/chat/completions`, the client normalizes it.

Azure CLI login is not needed for API-key inference. Use `az login` only when using Azure default credentials or when the Azure AI Evaluation SDK asks for project logging authentication.

`azd` is not needed for this repo's local benchmark runs. It is only needed for provisioning or hosted-agent workflows.

## Extension Points

Add a dataset by implementing a dataset adapter and registering it in `datasets/registry.py`.

Add a RAG system by implementing `RagSystem`, adding it to `systems/registry.py`, and giving it a config entry.

Add a model provider by implementing `TextGenerationClient` and wiring it in `models/factory.py`.

Add a workflow backend by implementing `WorkflowRunner` and selecting it through `workflow.runner`.

Add local metrics by extending `evaluation/local/` and listing the metric name in an experiment config.

Add a new report field by extending the typed report bundle in `reporting.py` and its focused tests.

## Final State

The repo is now ready for structured inspection milestone by milestone. The tracked source defines the reusable benchmark framework. Ignored artifacts preserve local run evidence without polluting git history.
