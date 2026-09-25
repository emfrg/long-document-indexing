# Repository Walkthrough

The [README](../README.md) explains the research question, document maps, main metrics,
and setup. This document follows one benchmark run through the repository so that you
can see where the data is loaded, where each retrieval system runs, and where the
results are written.

The main entry point is an experiment YAML file. The command-line program is called
`ldi`, short for Long Document Indexing. When you run `ldi run` with an experiment file,
the repository performs the following steps:

```text
experiment YAML
      |
      v
load cases and benchmark questions
      |
      v
build an index for every system and case
      |
      v
run every question through every system
      |
      v
save answers, routing decisions, and retrieved passages
      |
      v
calculate metrics and write reports
```

`ldi run` performs these stages in order. The separate `prepare`, `index`, `query`,
`evaluate`, and `report` commands expose the same lifecycle stages individually. Each
command takes the experiment file through `--config`.

The reference experiment is
[`foundry-multilexsum-legal-rag-qa-role-separated-extended.yaml`](../configs/experiments/foundry-multilexsum-legal-rag-qa-role-separated-extended.yaml).
It compares seven systems on 60 questions from 20 Multi-LexSum legal cases. The smaller
[`advanced-systems-smoke.yaml`](../configs/experiments/advanced-systems-smoke.yaml)
uses local test data and fake model clients, so it is the best place to start when you
want to inspect the workflow without making paid API calls.

## 1. The Experiment File

An experiment YAML file defines the complete comparison. The extended legal experiment
contains these sections:

| Section | What it controls |
| --- | --- |
| `experiment` | The experiment ID, random seed, and number of times each question is run. |
| `dataset` | The dataset version, selected cases, question file, and source-document chunking. |
| `models` | The Azure deployments used for map generation, routing, answering, judging, and embeddings. |
| `shared_pipeline` | How many documents the router may select and how many passages retrieval may return. |
| `systems` | The retrieval systems included in the comparison. |
| `run_control` | Resume behavior and the maximum number of model calls and tokens allowed. |
| `evaluation` | The local metrics and optional Foundry evaluators to run. |
| `storage` | The directory where the experiment writes its artifacts. |

For the reference experiment, the router may select up to three documents and the
dense retriever may return up to eight passages. Those limits apply equally to all six
document-map systems. The flat-vector baseline also returns up to eight passages.

The YAML is parsed and validated by
[`config.py`](../src/long_document_indexing/config.py). The model values in the
reference config resolve to Azure deployment names supplied through environment
variables, normally loaded from `.env`. See
[Run the Legal Benchmark](../README.md#run-the-legal-benchmark) and
[`.env.example`](../.env.example) for the required values.

The resolved configuration tells the dataset loader which cases and questions to load
in the next stage.

## 2. Cases and Questions

The dataset loader turns the source data into three simple levels:

```text
Corpus   = one legal case
Document = one source document in that case
Segment  = one passage cut from that document
```

The shared definitions are in
[`corpus.py`](../src/long_document_indexing/domain/corpus.py). The
[`MultiLexSumDatasetAdapter`](../src/long_document_indexing/datasets/multilexsum.py)
loads the fixed Multi-LexSum revision named in the experiment, selects the configured
cases, and creates stable document and segment IDs.

The 60 questions are stored in
[`rag-qa-extended.jsonl`](../benchmarks/multilexsum/rag-qa-extended.jsonl). Each line
contains one question and the information needed to score it later: the expected
answer, required documents, required passages, and supporting quotes. The
[case manifest](../benchmarks/multilexsum/rag-qa-extended-case-manifest.json) records
the 20 case IDs and the exact dataset revision used for their source documents.

The fields required for each loaded question are defined by `BenchmarkItem` in
[`benchmark.py`](../src/long_document_indexing/domain/benchmark.py). After
loading, the rest of the repository works with the same `Corpus` and `BenchmarkItem`
objects regardless of where the data came from.

The `Corpus` objects provide the documents to index. The `BenchmarkItem` objects provide
the questions to run and the labels used later for evaluation.

## 3. Models, Storage, and Retrieval

Before indexing begins, the CLI creates the resources used by every system:

- the map-generation, routing, and answer model clients;
- the embedding client and dense passage retriever;
- the artifact store used to save indexes, runs, and reports;
- the prompt loader, model-usage recorder, and progress reporter.

These resources are grouped in the `Services` object in
[`services.py`](../src/long_document_indexing/services.py). They are created from the
experiment configuration in
[`cli.py`](../src/long_document_indexing/cli.py). This gives every system the same
model roles, retrieval backend, storage rules, and prompt-loading mechanism for a
controlled comparison.

The indexing stage uses these services to build an `IndexArtifact` for each system
and case.

## 4. Building the Indexes

Every retrieval system implements the two operations defined by `RagSystem` in
[`base.py`](../src/long_document_indexing/systems/base.py):

1. `build_index` prepares one case for retrieval and returns an `IndexArtifact` that
   records where the resulting files were saved.
2. `run_query` answers one benchmark question using that case's saved index and returns
   a `RagRunRecord`.

For work that is not reused from an earlier run, the CLI calls `build_index` once for
every configured system and case. The two kinds of system build different artifacts.

### Flat-Vector Baseline

[`flat_vector.py`](../src/long_document_indexing/systems/flat_vector.py) embeds the raw
source passages and saves a dense index. It creates no document maps.

```text
source documents -> passages -> embeddings -> dense passage index
```

### Document-Map Systems

Each document-map system creates the same dense passage index and attempts to create one
map for every source document:

```text
source document -> passages -> map-building strategy -> document map
       |
       +---------------------> embeddings -> dense passage index
```

The six strategies differ in the way they construct the map:

| System | Map-construction method |
| --- | --- |
| `stuffing` | Sends the whole document to the map builder in one call. Documents above the context limit are recorded as `context_overflow`. |
| `map_reduce` | Maps document parts independently, then merges the partial maps. |
| `refine` | Reads passages in order and updates one evolving map after each passage. |
| `hierarchical_map` | Builds maps for small groups of passages, then combines them through several levels. |
| `outline_then_fill` | Creates an outline, fills its sections from the source passages, and assembles the final map. |
| `agentic_map` | Repeatedly inspects the document and revises the map until it reaches its stopping condition. |

Their implementations are in
[`systems/`](../src/long_document_indexing/systems). Their shared indexing and query
behavior is in
[`map_base.py`](../src/long_document_indexing/systems/map_base.py). Every completed map
uses the common `DocumentMap` and `MapEntry` structures defined in
[`maps.py`](../src/long_document_indexing/domain/maps.py).

Indexing results are saved after each case. Several map-building strategies also save
intermediate checkpoints while processing long documents. With resume enabled, the
repository checks that each saved artifact still exists and has the current index
signature, which is a fingerprint of the settings used to build it. The fingerprint
covers the retrieval backend, embedding model, map-building model and prompts, and
relevant configuration. A mismatch makes the artifact stale; use
`--allow-stale-recompute` only when replacing stale work is intentional. The separate
`--force` option rebuilds everything.

The saved `IndexArtifact` is passed to query execution so the next stage can reuse the
dense index and, for map-based systems, the document maps.

## 5. Running a Question

After every required index exists, the CLI calls `run_query` for each combination of
system and question.

### Flat-Vector Query

```text
question
   |
   v
dense search across every passage in the case
   |
   v
top 8 passages
   |
   v
answer model -> answer and citations
```

For the flat-vector baseline, document-level metrics use the first three distinct
document IDs appearing in its ranked list of up to eight retrieved passages. The
baseline does not run a separate document router.

### Document-Map Query

```text
question
   |
   v
router reads the document maps for the case
   |
   v
select up to 3 documents
   |
   v
dense search across passages from those documents
   |
   v
top 8 passages
   |
   v
answer model -> answer and citations
```

[`routing.py`](../src/long_document_indexing/routing.py) sends the complete maps and the
question to the routing model. The router returns the selected document IDs, a
rationale, and any unresolved information needs. The dense retriever then searches the
original passages belonging to those documents.

Map source references make each map entry traceable to its source passages and support
checks on map construction. The retrieved context comes from a separate dense search
after the router has selected documents.

Finally,
[`answering.py`](../src/long_document_indexing/answering.py) sends the retrieved source
passages to the answer model. The model returns a structured answer with citations, and
each citation must identify one of the retrieved evidence records. The code then assigns
the corresponding document and passage IDs to that citation.

Both query paths return the same `RagRunRecord` structure, described below.

## 6. The Standard Run Record

Every system writes the result of one question as a `RagRunRecord`, defined in
[`runs.py`](../src/long_document_indexing/domain/runs.py). It contains:

| Field group | Contents |
| --- | --- |
| Identity | Experiment, system, case, question, and repetition IDs. |
| Routing | Selected document IDs and the router's structured decision when a router was used. |
| Retrieval | Ranked passages, similarity scores, and source IDs. |
| Answer | Generated answer and citations. |
| Usage | Model calls, token counts, tool calls, and duration. |
| Status | Whether the question succeeded, failed, or was skipped, plus any error. |

The records are written to one JSONL file per system under `runs/`, with one JSON object
per line. A common record format allows the same evaluation and reporting code to
compare all seven systems.

## 7. Evaluation and Reporting

`ldi run` already performs local evaluation and reporting. Use `ldi evaluate` or
`ldi report` separately to recalculate metrics or regenerate reports from saved outputs.

`ldi evaluate` reads the run records and the expected answers, documents, passages, and
quotes from the question set. It calculates the configured local metrics directly,
without a judge model. It also reads the saved index artifacts when calculating
map-construction metrics. The metric implementations are in
[`local/`](../src/long_document_indexing/evaluation/local), and their common
runner is
[`runner.py`](../src/long_document_indexing/evaluation/runner.py).

The results address three parts of the pipeline:

1. **Document routing:** Did the system select the documents required by the question?
2. **Downstream retrieval:** Did the fixed passage retriever find the labeled evidence
   after document selection?
3. **Supporting checks:** Map validity, citation-quote matching, answer-reference wording
   overlap, runtime, and token use.

The README's
[What the Benchmark Measures](../README.md#what-the-benchmark-measures) section introduces
the routing and retrieval metrics. The [Metrics Reference](metrics.md) defines the full
set and distinguishes lexical comparisons from model-based answer evaluation.

When Foundry evaluation is enabled, `ldi evaluate` also exports the completed run
records as an evaluation dataset. For the complete extended experiment, this export
contains 420 rows. Two optional commands use that export for different purposes:

- `ldi evaluate-foundry-managed` runs the configured Foundry evaluators. These include
  model-based checks such as groundedness, relevance, and response completeness, and
  the label-based `document_retrieval` evaluator. The reference config evaluates the
  same fixed 20% sample for every system and checkpoints each evaluator separately.
- `ldi publish-foundry-evals` creates one portal-visible run per system and applies the
  deterministic retrieval, citation, and answer-overlap graders used for the Foundry
  comparison view.

Neither command reruns indexing, retrieval, or answer generation. They evaluate and
publish the saved benchmark outputs. The commands are documented in
[Optional Foundry Evaluation](../README.md#optional-foundry-evaluation).

`ldi report` aggregates the metric records and usage ledgers into Markdown, CSV, and
JSON summaries. The reporting code is in
[`reporting.py`](../src/long_document_indexing/reporting.py).

## 8. Where to Find the Outputs

Each experiment writes to `artifacts/<experiment-id>/`. After a complete reference run,
the most useful files are organized like this:

```text
artifacts/<experiment-id>/
  manifest.json                         resolved experiment and dataset details
  indexes/
    dense_vector/                       saved passage embeddings
    <system>/                           index metadata; map systems also store maps
  runs/
    <system>.jsonl                      one full result per question
  evaluations/
    local-metrics.jsonl                 individual deterministic metric values
    foundry/                            exports and optional Foundry results
  costs/
    index-usage.jsonl                   indexing tokens, calls, and duration
    query-usage.jsonl                   query tokens, calls, and duration
  workflows/                            per-case and per-question execution records
  report/
    results.md                          readable experiment report
    system-summary.csv                  one summary row per system
    results.csv                         mean value for every system and metric
    confidence-intervals.csv            metric means with 95% intervals
    usage-summary.csv                   token, call, and duration totals
    summary.json                        complete machine-readable report
```

Start with `report/results.md` for the comparison. Open `system-summary.csv` for one row
per system, and `results.csv` for every metric. To investigate a specific result, find
the question in `runs/<system>.jsonl`, then inspect the selected documents, retrieved
passages, answer, and citations stored in that record.

## 9. Running the Repository

After completing the installation steps in
[Quick Start](../README.md#quick-start), run all seven systems with local fixtures
and fake model clients:

```bash
uv run --python 3.13 \
  --no-editable \
  --reinstall-package long-document-indexing \
  ldi run \
  --config configs/experiments/advanced-systems-smoke.yaml
```

This run checks the full software workflow. Use its scores to confirm expected test
behavior; the Multi-LexSum experiment provides the retrieval comparison.

After completing the setup in
[Run the Legal Benchmark](../README.md#run-the-legal-benchmark), preview the budget
before making model calls:

```bash
uv run --python 3.13 \
  --extra foundry \
  --extra multilexsum \
  --no-editable \
  --reinstall-package long-document-indexing \
  ldi run \
  --config configs/experiments/foundry-multilexsum-legal-rag-qa-role-separated-extended.yaml \
  --dry-run-budget
```

The setup, role-separated smoke test, and extended run commands are in
[Run the Legal Benchmark](../README.md#run-the-legal-benchmark). The additional evaluator
and publication commands are in
[Optional Foundry Evaluation](../README.md#optional-foundry-evaluation).

You can also run the lifecycle phases separately with `ldi prepare`, `ldi index`,
`ldi query`, `ldi evaluate`, and `ldi report` when investigating one stage. Use the same
`--config` path to work with the same experiment and its saved artifacts.

## 10. Finding the Code You Need

| To understand or change... | Start here |
| --- | --- |
| Experiment settings | [`experiments/`](../configs/experiments) |
| Per-system settings | [`systems/`](../configs/systems) |
| Multi-LexSum loading and chunking | [`multilexsum.py`](../src/long_document_indexing/datasets/multilexsum.py) |
| Corpus, question, map, and run data definitions | [`domain/`](../src/long_document_indexing/domain) |
| A map-construction strategy | [`systems/`](../src/long_document_indexing/systems) |
| Shared map routing and retrieval | [`map_base.py`](../src/long_document_indexing/systems/map_base.py) |
| Router requests and responses | [`routing.py`](../src/long_document_indexing/routing.py) |
| Dense passage indexing and search | [`dense_vector.py`](../src/long_document_indexing/retrieval/dense_vector.py) |
| Answer generation and citation validation | [`answering.py`](../src/long_document_indexing/answering.py) |
| Deterministic metrics | [`local/`](../src/long_document_indexing/evaluation/local) |
| Foundry evaluation and publication | [`foundry/`](../src/long_document_indexing/evaluation/foundry) |
| Report generation | [`reporting.py`](../src/long_document_indexing/reporting.py) |
| Command-line orchestration | [`cli.py`](../src/long_document_indexing/cli.py) |

To add a dataset, implement a loader that returns the shared corpus and question
objects, then register it in
[the dataset registry](../src/long_document_indexing/datasets/registry.py). To add a
retrieval system, implement the two `RagSystem` operations and register it in
[the system registry](../src/long_document_indexing/systems/registry.py). Keeping
those shared input and output structures allows the existing query, evaluation, and
reporting code to include the new implementation.
