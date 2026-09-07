# Long Document Indexing

A Python framework for benchmarking long-document indexing strategies for retrieval-augmented generation.

The repository is intentionally benchmark-first:

1. Normalize corpora and benchmark questions into stable schemas.
2. Build one or more RAG system indexes.
3. Run questions through each system.
4. Persist standardized run records.
5. Evaluate deterministic retrieval and citation metrics locally.
6. Add Microsoft Agent Framework and Foundry evaluation behind adapters later.

The first milestone is local-only. It includes the canonical domain model, a JSONL smoke dataset, a simple lexical retrieval backend, a flat-vector-style baseline, and local metrics.

Python 3.12 or 3.13 is required. Python 3.14 release candidates are intentionally excluded until the dependency stack supports them cleanly.

## Quick Start

```bash
uv sync --python 3.13 --extra dev
uv run --python 3.13 --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/smoke-test.yaml
uv run --python 3.13 --extra dev pytest
```

The smoke run writes artifacts under `artifacts/`, which is ignored by git.

The CLI command uses `--no-editable` because some macOS Python 3.13 environments skip editable-install `.pth` files inside hidden virtualenv directories. The explicit package reinstall keeps the console script aligned with local source changes while the repo is under active development. Tests still run against `src/` through pytest configuration.

## Current Shape

Milestone 1 established the benchmark kernel. Milestone 2 adds workflow-neutral orchestration, local workflow run records, trace IDs, usage ledgers, prompt loading, and an optional Microsoft Agent Framework adapter boundary.

Milestone 3 added the first document-map systems: `stuffing`, `map_reduce`, and `refine`. These systems use the shared `TextGenerationClient` interface, with a deterministic fake implementation for local tests and smoke runs.

Milestone 5 added an optional Microsoft Agent Framework functional workflow runner. The concrete default smoke path still runs locally.

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

## MAF Workflow Runner

Milestone 5 adds an optional Microsoft Agent Framework functional workflow runner. It does not require Azure resources when used with the fake generator.

```bash
uv sync --python 3.13 --extra dev --extra maf
uv run --python 3.13 --extra maf --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/maf-stuffing-smoke.yaml
```
