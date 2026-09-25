# Extended Multi-LexSum Benchmark Results

This directory is a versioned snapshot of the generated report for the experiment
`foundry-multilexsum-legal-rag-qa-role-separated-extended`. These are the results
reported in the accompanying article about long-document indexing for RAG.

The complete local artifact tree is not included. It is approximately 180 MB and
contains generated indexes, maps, run records, workflow checkpoints, and copied source
passages. This snapshot keeps the compact aggregate reports needed to inspect the
published comparisons.

## Experiment

- Configuration: [`configs/experiments/foundry-multilexsum-legal-rag-qa-role-separated-extended.yaml`](../../configs/experiments/foundry-multilexsum-legal-rag-qa-role-separated-extended.yaml)
- Questions: [`benchmarks/multilexsum/rag-qa-extended.jsonl`](../../benchmarks/multilexsum/rag-qa-extended.jsonl)
- Case manifest: [`benchmarks/multilexsum/rag-qa-extended-case-manifest.json`](../../benchmarks/multilexsum/rag-qa-extended-case-manifest.json)
- Dataset revision: `055f5fa5300a5fdefb5b691c4d6c1e58f485b379`
- Cases: 20
- Questions: 60
- Systems: 7
- Answer runs: 420
- Query repetitions: 1
- Seed: 42
- Report generated: 2026-09-15

The generated experiment manifest recorded these model roles:

| Role | Deployment |
| --- | --- |
| Map builder | `gpt-5-mini-doc-map-generator` |
| Router | `gpt-5-mini` |
| Answerer | `gpt-5-mini` |
| Embedding | `text-embedding-3-large` |
| Judge | `gpt-5` |

The generated manifest did not record a source-code commit. The dataset revision,
experiment configuration, question set, case manifest, seed, model roles, and aggregate
outputs are preserved here so the reported run remains inspectable.

## Files

| File | Contents |
| --- | --- |
| [`results.md`](results.md) | Human-readable scorecard, metric tables, usage summary, and reported issues. |
| [`results.csv`](results.csv) | Mean value and count for every system and metric. |
| [`confidence-intervals.csv`](confidence-intervals.csv) | Metric means with 95% confidence intervals. |
| [`system-summary.csv`](system-summary.csv) | One aggregate comparison row per system. |
| [`usage-summary.csv`](usage-summary.csv) | Model calls, tokens, tool calls, and duration by system and phase. |
| [`summary.json`](summary.json) | Complete machine-readable aggregate report. |

The report files are copied from
`artifacts/foundry-multilexsum-legal-rag-qa-role-separated-extended/report/`. The only
sanitization is replacement of the machine-local absolute Foundry manifest path with
the corresponding repository-relative artifact path in `results.md` and `summary.json`.
