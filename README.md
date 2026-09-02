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

The concrete smoke path still runs locally. The MAF runner is intentionally a boundary for now; model-backed MAF workflows arrive after the local benchmark contracts are stable.
