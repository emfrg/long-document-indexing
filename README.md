# Long Document Indexing

A Python framework for benchmarking long-document indexing strategies for retrieval-augmented generation.

The repository is intentionally benchmark-first:

1. Normalize corpora and benchmark questions into stable schemas.
2. Build one or more RAG system indexes.
3. Run questions through each system.
4. Persist standardized run records.
5. Evaluate deterministic retrieval, evidence, and citation metrics locally.
6. Export Foundry-ready evaluation datasets behind an adapter.

The first milestone is local-only. It includes the canonical domain model, a JSONL smoke dataset, a simple lexical retrieval backend, a flat-vector-style baseline, and local metrics.

Python 3.12 or 3.13 is required. Python 3.14 release candidates are intentionally excluded until the dependency stack supports them cleanly.

## Repo Walkthrough

For the architectural walkthrough, milestone map, runtime flow, and extension points, read [docs/repo-walkthrough.md](docs/repo-walkthrough.md).

The short version is:

```text
experiment config -> dataset adapter -> services -> index/query workflows
                  -> RAG system -> run records + usage -> metrics/export/report
```

## Milestone Map

| Milestone | Name | Boundary |
| --- | --- | --- |
| 1 | Python benchmark kernel | Local domain schemas, smoke data, lexical retrieval, baseline metrics. |
| 2 | Workflow and Telemetry foundation | Workflow runner boundary, trace records, usage ledgers, prompt loading. |
| 3 | First Document Maps Systems | `stuffing`, `map_reduce`, and `refine` systems over shared map contracts. |
| 4 | Real Model Client thin slice | OpenAI-compatible inference client behind the generation interface. |
| 5 | MAF workflow thin slice | Optional Microsoft Agent Framework runner without changing system code. |
| 6 | Foundry GPT-5 Structured Generation | Responses API path with Pydantic-backed structured outputs. |
| 7 | Model-Backed Answer Generation | Optional generated answers with validated retrieved-evidence citations. |
| 8 | Foundry Evaluation Export | Local JSONL export in a Foundry-ready single-turn shape. |
| 9 | Multi-System Real Benchmark Run | Real GPT-5-family smoke comparison across all implemented systems. |
| 10 | Analysis And Reporting Upgrade | Markdown, CSV, usage, issue, and system scorecard reports. |
| 11 | Larger Benchmark Dataset Thin Slice | Versioned synthetic enterprise benchmark fixture. |
| 12 | Resumable And Budgeted Live Runs | Resume, force, dry-run budget, and model-call/token/cost limits. |
| 13 | Foundry Managed Evaluation Execution | Azure AI Evaluation SDK path over exported datasets. |
| 14 | Larger Real Benchmark Run | GPT-5-family stuffing run over the enterprise thin slice. |
| 15 | Repo Polish And Final Walkthrough | Documentation, verification, and final repository tour. |

## Quick Start

```bash
uv sync --python 3.13 --extra dev
uv run --python 3.13 --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/smoke-test.yaml
uv run --python 3.13 --extra dev pytest
```

The smoke run writes artifacts under `artifacts/`, which is ignored by git.

The CLI command uses `--no-editable` because some macOS Python 3.13 environments skip editable-install `.pth` files inside hidden virtualenv directories. The explicit package reinstall keeps the console script aligned with local source changes while the repo is under active development. Tests still run against `src/` through pytest configuration.

## Current Shape

The default path is still local and deterministic: `generator_provider: fake` runs without Azure credentials and exercises the same orchestration, metric, reporting, and export code used by real-model experiments.

The real-model path is optional. Foundry/Azure OpenAI inference is isolated behind `TextGenerationClient`, and GPT-5-family structured generation uses the Responses API with Pydantic output contracts. API-key inference does not require Azure CLI login.

The Foundry evaluation path is split deliberately. Local export writes JSONL datasets under ignored `artifacts/`; managed evaluation uses the Azure AI Evaluation SDK when requested. Logging managed results to a Foundry project requires a project endpoint and may require `az login`. `azd` is only needed for provisioning or hosted-agent workflows.

The RAG evaluation path is intentionally concrete: use labeled questions with
`relevant_document_ids`, `relevant_segment_ids`, and evidence quotes. Local metrics then
score route quality, retrieved context quality, evidence quote recovery, citation target
validity, and citation quote support. ROUGE/F1-style reference overlap remains useful as a
secondary answer check, but it is not the primary RAG signal.

## Implemented Systems

The repo now has seven comparable systems:

| System | Config id | Role |
| --- | --- | --- |
| Flat vector baseline | `flat_vector` | Raw segment retrieval without document maps. |
| Stuffing map | `stuffing` | One map from the full document when it fits the context budget. |
| Map-reduce map | `map_reduce` | Segment maps reduced into one document map. |
| Refine map | `refine` | Sequential map revision over ordered segments. |
| Hierarchical map | `hierarchical_map` | Leaf maps reduced through a bounded hierarchy. |
| Outline-then-fill map | `outline_then_fill` | Plan an outline, then fill outline nodes from source segments. |
| Agentic map | `agentic_map` | Bounded inspect-and-revise loop with coverage metadata. |

Run all seven on the local smoke fixture:

```bash
uv run --python 3.13 --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/advanced-systems-smoke.yaml
```

Run all seven on the synthetic enterprise fixture:

```bash
uv run --python 3.13 --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/enterprise-advanced-thin-slice.yaml
```

## Real Model Client

Milestone 4 adds a thin OpenAI-compatible client for Microsoft Foundry/Azure OpenAI inference endpoints. Milestone 6 adds a Responses API path with Pydantic-backed structured outputs for GPT-5-family deployments. The real client is optional; the default smoke test still uses `generator_provider: fake`.

Install the optional dependencies only when running against a real deployment:

```bash
uv sync --python 3.13 --extra dev --extra foundry
```

Set real model values through environment variables or a local `.env` file:

```bash
export FOUNDRY_GENERATOR_BASE_URL="https://<resource>.openai.azure.com/openai/v1/"
export FOUNDRY_GENERATOR_DEPLOYMENT="<deployment-name>"
export FOUNDRY_GENERATOR_API="responses"
export FOUNDRY_GENERATOR_MAX_OUTPUT_TOKENS="2000"
export FOUNDRY_GENERATOR_RESPONSE_FORMAT="structured"
export AZURE_INFERENCE_CREDENTIAL="<api-key>"
```

`FOUNDRY_GENERATOR_BASE_URL` should be the base `/openai/v1/` endpoint. If a copied endpoint ends in `/responses` or `/chat/completions`, the client normalizes it back to the base URL before creating the SDK client.

`FOUNDRY_GENERATOR_RESPONSE_FORMAT=structured` is the recommended GPT-5 path. It uses the OpenAI SDK `responses.parse(..., text_format=...)` flow so Pydantic generates the schema and parses the result. `json_object` remains available only as a compatibility fallback.

With `generator_auth_mode: api_key`, Azure CLI login is not required. Azure login is only needed if you switch the config to `generator_auth_mode: azure_default_credential` or start provisioning/managing Azure resources.

Real-model configs include `run_control.resume: true` and a conservative `max_model_calls` cap. Completed index artifacts and successful query records are reused on rerun. Use `--force` only when you intentionally want to spend calls again.

Then run the real-client stuffing smoke config:

```bash
uv run --python 3.13 --extra foundry --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/foundry-stuffing-smoke.yaml
```

## Answer Generation

The default query path is extractive: it builds answers by stitching retrieved source snippets. This keeps local smoke tests deterministic.

Milestone 7 adds optional model-backed answer generation:

```yaml
answering:
  mode: generated
```

Generated answers use the same `TextGenerationClient` and Pydantic structured-output flow as document-map generation. The model returns answer text plus citations, and the repo validates that every citation points to retrieved evidence.

Run the generated-answer smoke config after configuring the real model `.env` values:

```bash
uv run --python 3.13 --extra foundry --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/foundry-generated-answer-smoke.yaml
```

## Foundry Evaluation Export

Milestone 8 exports benchmark run records into a Foundry-ready JSONL dataset. The export includes the standard single-turn fields `query`, `response`, `context`, and `ground_truth`, plus inspectable metadata for systems, citations, retrieved chunks, selected documents, and document-retrieval labels.

Enable automatic export during `ldi evaluate`:

```yaml
evaluation:
  foundry:
    enabled: true
    dataset_path: evaluations/foundry/dataset.jsonl
    manifest_path: evaluations/foundry/manifest.json
    evaluation_level: turn
```

You can also export from existing run artifacts:

```bash
uv run --python 3.13 --no-editable --reinstall-package long-document-indexing ldi export-foundry-eval --config configs/experiments/foundry-eval-export-smoke.yaml
```

This milestone does not run a cloud evaluation. Azure login is not required for the local export.

## Foundry Managed Evaluation

Milestone 13 adds a thin Azure AI Evaluation SDK execution path over the exported JSONL dataset. Inspect the exact SDK call shape first:

```bash
uv run --python 3.13 --extra foundry --no-editable --reinstall-package long-document-indexing ldi evaluate-foundry-managed --config configs/experiments/enterprise-thin-slice.yaml --dry-run
```

The dry run writes `evaluations/foundry/managed-plan.json` and makes no cloud call. To log results to a Foundry project, set the project endpoint:

```bash
export FOUNDRY_EVALUATION_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project-name>"
uv run --python 3.13 --extra foundry --no-editable --reinstall-package long-document-indexing ldi evaluate-foundry-managed --config configs/experiments/enterprise-thin-slice.yaml
```

The default managed evaluators are `f1` and `rouge`, which score `response` against `ground_truth`. For RAG-labeled datasets, configure managed evaluators such as `groundedness`, `relevance`, `retrieval`, `document_retrieval`, and `response_completeness`; these consume the exported `query`, `response`, `context`, `retrieved_documents`, `retrieval_ground_truth`, and `ground_truth` fields. Model-judge evaluators use `models.judge_deployment` when set, otherwise `models.generator_deployment`, plus the normalized `models.generator_base_url`.

Use `.env` for local settings:

```bash
FOUNDRY_EVALUATION_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project-name>"
FOUNDRY_GENERATOR_BASE_URL="https://<resource>.services.ai.azure.com/openai/v1/"
FOUNDRY_GENERATOR_DEPLOYMENT="<deployment-name>"
AZURE_INFERENCE_CREDENTIAL="<api-key>"
# Optional override when the judge deployment is a GPT-5/o-series reasoning model.
FOUNDRY_EVALUATION_REASONING_MODEL="true"
```

When `models.judge_deployment` or `models.generator_deployment` looks like a GPT-5/o-series deployment, the managed-evaluation adapter marks Azure AI Evaluation SDK model judges as reasoning models. This makes the SDK use the parameter shape those models require. Set `FOUNDRY_EVALUATION_REASONING_MODEL=true` or `false` in `.env` only when you need to override auto-detection.

The SDK's batch logs use "lines" to mean dataset rows. For example, `Finished 22 / 28 lines` means one evaluator has scored 22 of 28 exported rows. If multiple evaluator names advance at the same timestamp, the SDK is running evaluator batches concurrently.

For RAG judge runs on adequately provisioned judge deployments, use concurrent evaluator execution:

```yaml
evaluation:
  foundry:
    managed_execution: parallel
    managed_evaluator_delay_seconds: 0
```

This calls the Azure AI Evaluation SDK once with all configured evaluators. It is the faster path and matches the default RAG smoke configuration.

For RAG judge runs on low quota deployments, switch to sequential evaluator execution:

```yaml
evaluation:
  foundry:
    managed_execution: sequential
    managed_evaluator_delay_seconds: 5
```

Sequential execution calls the SDK once per evaluator and merges the resulting metrics into the normal managed-result artifact. This reduces the largest request burst, though the SDK may still parallelize rows inside a single evaluator. If the SDK asks for Azure authentication when logging to a project, run `az login`. `azd` is only needed for provisioning or Foundry hosted-agent workflows, not for local export or dry-run planning. GPT-5-family judge runs can still hit Azure rate limits; the SDK retries, but low quota can make managed evaluation slower.

## Foundry Portal Evals

The Azure AI Evaluation SDK can upload completed evaluation results, but those uploaded runs may not appear as top-level rows in the Foundry Evaluations page. To create a portal-visible Foundry Evals parent and run from the exported JSONL dataset, use:

```bash
uv run --python 3.13 --extra foundry --no-editable --reinstall-package long-document-indexing ldi publish-foundry-evals --config configs/experiments/enterprise-thin-slice.yaml --dry-run
uv run --python 3.13 --extra foundry --no-editable --reinstall-package long-document-indexing ldi publish-foundry-evals --config configs/experiments/enterprise-thin-slice.yaml
```

This command reads `FOUNDRY_EVALUATION_PROJECT_ENDPOINT` from `.env`, uses the current Azure CLI login through `DefaultAzureCredential`, uploads an Evals-shaped JSONL file, and creates a Foundry Evals run. The command writes `evaluations/foundry/openai-evals-result.json`, including the Foundry `report_url`.

Use this portal-visible path whenever the goal is to inspect results in the Foundry Evaluations UI. It maps configured `managed_evaluators` to native Evals criteria: `f1` becomes deterministic token-F1, `rouge` becomes ROUGE-1 text similarity, and RAG evaluator names become deterministic Python graders over the exported RAG fields:

- `groundedness` -> `citation_support_rate`
- `relevance` -> `answer_reference_token_f1`
- `retrieval` -> `context_precision_at_4`, `context_recall_at_4`, `evidence_quote_recall_at_4`
- `document_retrieval` -> `document_retrieval_precision`, `document_retrieval_recall`
- `response_completeness` -> `answer_reference_token_recall`

Keep `ldi evaluate-foundry-managed` for backend SDK scoring and local artifacts. Do not rely on it for Foundry UI visibility.

For benchmark presentation, publish one Foundry run per system so the Foundry run table compares systems directly:

```bash
uv run --python 3.13 --extra foundry --extra multilexsum --no-editable --reinstall-package long-document-indexing ldi publish-foundry-evals --config configs/experiments/foundry-multilexsum-legal-rag-qa-smoke.yaml --evaluation-name ldi-multilexsum-legal-rag-qa-system-comparison --run-name legal-rag-smoke
```

This creates one run per `system_id`, named like `legal-rag-smoke-map_reduce`, `legal-rag-smoke-refine`, and so on. The Foundry publishing path is intentionally system-by-system because the benchmark exists to compare long-document indexing strategies.

## Multi-System Real Benchmark

Milestone 9 runs the real GPT-5-family generator across all implemented systems on the smoke corpus:

```bash
uv run --python 3.13 --extra foundry --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/foundry-multi-system-real-smoke.yaml
```

This config compares `flat_vector`, `stuffing`, `map_reduce`, and `refine` with `answering.mode: generated` and Foundry evaluation export enabled. On the bundled smoke corpus, it performs 12 document-map generation calls and 8 generated-answer calls. The run writes local metrics, usage JSONL files, a Markdown/CSV report, and a Foundry-ready evaluation dataset under `artifacts/foundry-multi-system-real-smoke/`.

Azure login is not required for this API-key based run. Configure `FOUNDRY_GENERATOR_BASE_URL`, `FOUNDRY_GENERATOR_DEPLOYMENT`, and `AZURE_INFERENCE_CREDENTIAL` before running it.

## Multi-LexSum Legal Benchmark

The legal branch adds Multi-LexSum case-file support for longer, multi-document legal corpora. The selected thin slice uses 5 Civil Rights Litigation Clearinghouse cases, 32 source documents, and 186 source segments.

Run the resumable development benchmark:

```bash
uv run --python 3.13 --extra foundry --extra multilexsum --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/foundry-multilexsum-legal-thin-slice.yaml
```

Run the clean all-systems benchmark for blog/report tables:

```bash
uv run --python 3.13 --extra foundry --extra multilexsum --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/foundry-multilexsum-legal-clean-all-systems.yaml
```

Both configs use API-key auth, so Azure CLI login is not required. They enable resume and checkpoint reuse because real legal runs can span many model calls. Map artifacts record prompt-safety and source-reference normalization policy versions; stale map artifacts are rebuilt instead of silently reused.

For proper legal RAG evaluation, use the curated QA overlay:

```bash
uv run --python 3.13 --extra foundry --extra multilexsum --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/foundry-multilexsum-legal-rag-qa-smoke.yaml
```

This config uses evidence-labeled questions rather than whole-case summary prompts, so
its report includes context precision/recall, evidence quote recall, citation precision,
citation recall, and citation support rate.

## Enterprise Thin Slice Benchmark

Milestone 11 adds a larger local benchmark fixture under `benchmarks/enterprise/`. It is synthetic and versioned in the repo: 8 documents, 24 source segments, and 16 gold-labeled questions covering policy, incident, launch, budget, training, audit, and escalation notes.

Run the deterministic all-system slice without Azure credentials:

```bash
uv run --python 3.13 --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/enterprise-thin-slice.yaml
```

This compares `flat_vector`, `stuffing`, `map_reduce`, and `refine` with generated answers from the deterministic fake generator, local metrics, usage summaries, report files, and a Foundry-ready evaluation export.

After configuring the real model `.env` values, run the optional GPT-5-family stuffing-only slice:

```bash
uv run --python 3.13 --extra foundry --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/foundry-enterprise-stuffing-thin-slice.yaml
```

That config is intentionally limited to `stuffing` to cap live model calls while exercising the same larger dataset and export path. With API-key auth, Azure login is not required.

## Resumable And Budgeted Runs

Milestone 12 adds run-control safety for real-model experiments:

```yaml
run_control:
  resume: true
  max_model_calls: 24
  max_total_tokens: 50000
```

`resume: true` reuses valid index artifacts and successful query records. Failed or skipped query records are retried. `max_model_calls`, `max_input_tokens`, `max_output_tokens`, `max_total_tokens`, and `max_estimated_cost` are enforced against recorded usage.

CLI flags can override the config for one command:

```bash
uv run --python 3.13 --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/enterprise-thin-slice.yaml --resume
uv run --python 3.13 --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/enterprise-thin-slice.yaml --force
uv run --python 3.13 --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/enterprise-thin-slice.yaml --dry-run-budget
```

The runner writes index artifacts, run records, and usage ledgers incrementally. If a budget is exceeded after a completed unit, the completed artifact remains on disk and the next run can resume from that point.

## Reporting

`ldi report` writes a compact analysis bundle under each experiment's `report/` directory:

```text
report/results.md
report/results.csv
report/system-summary.csv
report/usage-summary.csv
report/summary.json
```

`results.csv` remains the simple metric mean table for compatibility. `system-summary.csv` adds quality, routing, retrieval, answer, map, latency, token, model-call, and issue columns. `usage-summary.csv` aggregates index/query usage by system and event kind. `summary.json` preserves the complete typed report bundle for scripts or notebooks.

## MAF Workflow Runner

Milestone 5 adds an optional Microsoft Agent Framework functional workflow runner. It does not require Azure resources when used with the fake generator.

```bash
uv sync --python 3.13 --extra dev --extra maf
uv run --python 3.13 --extra maf --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/maf-stuffing-smoke.yaml
```
