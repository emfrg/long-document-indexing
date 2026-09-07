# Enterprise Thin Slice Benchmark

This benchmark is a hand-authored local dataset for exercising the benchmark pipeline beyond the two-question smoke corpus.

It contains:

- 8 synthetic enterprise documents.
- 24 canonical source segments.
- 16 benchmark questions.
- Gold expected answers.
- Relevant document and segment labels for local routing, retrieval, and citation metrics.

The dataset is intentionally small and versioned in the repo. It is not meant to model a production distribution yet; it exists to expose cross-document routing, retrieval, citation, and reporting behavior before larger external datasets or live Foundry evaluation runs are introduced.

Use `configs/experiments/enterprise-thin-slice.yaml` for the local deterministic all-system run. Use `configs/experiments/foundry-enterprise-stuffing-thin-slice.yaml` only after credentials are configured; it is resumable and capped at 24 model calls.
