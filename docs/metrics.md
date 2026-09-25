# Metrics Reference

This reference defines the metrics used by the extended Multi-LexSum comparison. The
[README](../README.md#what-the-benchmark-measures) introduces the benchmark and its main
retrieval metrics. The [Repository Walkthrough](repo-walkthrough.md#7-evaluation-and-reporting)
explains how evaluation reads the saved run records and produces reports.

`ldi run` computes the configured local metrics and writes the report. Foundry
evaluators run separately through the optional commands documented in
[Optional Foundry Evaluation](../README.md#optional-foundry-evaluation).

## Primary Routing Metrics

These metrics directly measure whether document maps help select the required documents.

For the flat-vector baseline, document-level metrics use the first three distinct
document IDs appearing in its ranked list of up to eight retrieved passages. The
baseline does not run a separate document router.

| Metric | What it measures |
| --- | --- |
| `document_recall_at_1` | The fraction of required documents found in the first document choice. |
| `document_recall_at_3` | The fraction of required documents found in the first three choices. |
| `mrr` | How highly the first required document was ranked. |
| `required_document_coverage` | Whether every required document was selected. |

`required_document_coverage` is 1 when every required document is selected and 0
otherwise. Its mean is the fraction of questions for which the system selected all
required documents. Reciprocal rank is the inverse of the first relevant document's
rank, or 0 when no relevant document was selected; `mrr` averages this across questions.

The implementations are in
[`routing.py`](../src/long_document_indexing/evaluation/local/routing.py).

## Downstream Retrieval Metrics

Every document-map system uses the same dense passage retriever. These metrics show how
the routing decision affects the passages that retriever finds. The reference experiment
allows up to eight retrieved passages, while the configured local metrics below inspect
the first four.

| Metric | What it measures |
| --- | --- |
| `segment_recall_at_4` | The fraction of required passages found in the first four results. |
| `context_precision_at_4` | Whether relevant passages were ranked ahead of irrelevant passages in the first four results. |
| `context_recall_at_4` | The fraction of required passages found in the first four results. |
| `evidence_quote_recall_at_4` | The fraction of gold evidence quotes found in the first four passages, after normalizing case and whitespace. |

Context precision is rank-aware: irrelevant results above relevant results lower the
score. It is not simply the fraction of the four results that are relevant.

Context recall uses passage labels when they are available and document labels
otherwise. The extended benchmark has passage labels, so `context_recall_at_4` and
`segment_recall_at_4` measure the same coverage here. Evidence-quote recall instead
checks whether the labeled quotes occur in the retrieved text after normalizing case
and whitespace.

The implementations are in
[`rag.py`](../src/long_document_indexing/evaluation/local/rag.py) and
[`routing.py`](../src/long_document_indexing/evaluation/local/routing.py).

## Map Construction

These metrics describe the saved document maps, their source references, and completion.

| Metric | What it measures |
| --- | --- |
| `map_schema_validity` | The fraction of saved maps that can be loaded as valid document maps. |
| `map_source_reference_validity` | The fraction of source references that point to existing passages in the correct document. |
| `map_compression_ratio` | The estimated size of the map text relative to the source passage text. |
| `map_completion_rate` | The fraction of source documents for which a map can be loaded. |

Source-reference validity checks that the referenced locations exist and belong to the
correct document. It does not establish that every map entry accurately represents its
source. Similarly, completion records the presence of a loadable map, not full coverage
of every passage or fact.

`map_compression_ratio` compares approximate token counts for map text and source
passage text. It is a size ratio, not a quality score or a measurement of the complete
router request. A ratio above 1 is possible.

The implementations are in
[`maps.py`](../src/long_document_indexing/evaluation/local/maps.py).

## Citations

| Metric | What it measures |
| --- | --- |
| `citation_precision` | The fraction of generated citations that point to required documents or passages. |
| `citation_recall` | The fraction of required documents or passages covered by the generated citations. |
| `citation_support_rate` | The fraction of generated citations whose quote is found in the source location they cite. |
| `invalid_citation_rate` | The fraction of citations that point to an unknown document or passage. |

Citation precision and recall compare citations with the benchmark labels.
`citation_support_rate` checks whether a citation's quote occurs in the cited source
after normalizing case and whitespace. This is a text-matching check, not a semantic
judgment that the quote supports every claim in the answer.

The implementations are in
[`answers.py`](../src/long_document_indexing/evaluation/local/answers.py) and
[`rag.py`](../src/long_document_indexing/evaluation/local/rag.py).

## Answer Comparison

| Metric | What it measures |
| --- | --- |
| `answer_reference_token_precision` | How much of the generated answer's wording overlaps the gold-standard answer. |
| `answer_reference_token_recall` | How much of the gold-standard answer's wording appears in the generated answer. |
| `answer_reference_token_f1` | The balance between answer token precision and recall. |

These are lexical comparisons with the reference answer. They do not independently
establish factual correctness or semantic completeness. The optional Foundry evaluators
provide separate model-based assessments of groundedness and response completeness.

The implementations are in
[`answers.py`](../src/long_document_indexing/evaluation/local/answers.py).

## Runtime and Usage

| Metric | What it measures |
| --- | --- |
| `query_duration_ms` | Time spent processing one question. |
| `tool_calls` | Query-stage operations: retrieval for the baseline, and routing plus retrieval for map-based systems. |

The usage report separately records model calls and input and output tokens for indexing
and querying. `tool_calls` counts the query-stage operations described above; it is not
the number of model API calls.

## Foundry Evaluators

| Evaluator | What it measures |
| --- | --- |
| `groundedness` | Whether the answer is supported by the retrieved context. |
| `relevance` | Whether the answer addresses the question. |
| `retrieval` | The quality of the retrieved context. |
| `document_retrieval` | Whether the selected documents match the required documents. |
| `response_completeness` | Whether the answer covers the information in the gold-standard answer. |

`document_retrieval` compares retrieval results with ground-truth relevance labels; it
does not use an LLM judge. The other evaluators listed here are model-based. Foundry
evaluators use their own scoring scales.

The reference configuration evaluates the same deterministic 20% sample for every
system. This is separate from publishing the full set of deterministic benchmark
metrics as one portal-visible run per system.

The integration is in
[`foundry/`](../src/long_document_indexing/evaluation/foundry).

## Reading the Scores

The local routing, retrieval, map-validity, map-completion, citation, and answer-overlap
scores range from 0 to 1. Higher is better, except for `invalid_citation_rate`.
`map_compression_ratio`, runtime, and usage describe trade-offs rather than quality.

Start with `report/results.md` for the comparison, `report/results.csv` for metric
means, and `report/usage-summary.csv` for model usage. To investigate an individual
score, inspect the corresponding question in `runs/<system>.jsonl` and its metric
records in `evaluations/local-metrics.jsonl`.
