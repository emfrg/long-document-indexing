# Long Document Indexing

This repository studies whether **document maps** can help a retrieval-augmented
generation (RAG) system identify which documents should be searched for evidence. The
reference experiment uses Multi-LexSum legal case files and compares a dense-retrieval
baseline with six document-map strategies. Every system answers the same questions,
uses the same answer model, and is scored against the same retrieval labels.

Legal research often involves a large number of multi-page documents. A case file can
grow to thousands of pages over time and may contain complaints, motions, exhibits,
court orders, and other material. Importantly, these documents describe different
stages of the same matter, often using different language for the same people, events,
and legal issues.

A retrieval system that provides useful answers may therefore depend on evidence spread
across the case file. For example, a question could require connecting an allegation in
a complaint with a later court ruling and the remedy recorded in a settlement (a
multi-hop question). The relevant evidence is distributed across passages and documents
that play different roles in the case.

The reference experiment uses Azure OpenAI deployments configured through Microsoft
Foundry for map generation, routing, answer generation, embeddings, and model-based
evaluation. A deterministic local smoke test is also included; it requires no Azure
credentials or paid API calls.

The central research question is:

> Does a structured map of each document help a RAG system route complex questions to
> the documents that contain the required evidence?

[Quick Start](#quick-start) ·
[How It Works](#how-map-based-rag-works-in-this-repository) ·
[Run the Legal Benchmark](#run-the-legal-benchmark) ·
[Outputs](#where-to-find-the-output) ·
[Metrics](#what-the-benchmark-measures) ·
[Repository Walkthrough](docs/repo-walkthrough.md)

## Quick Start

Run these commands from the root of a local checkout of this repository.

Requirements:

- Python 3.12 or 3.13;
- [`uv`](https://docs.astral.sh/uv/).

Install the development dependencies:

```bash
uv sync --python 3.13 --extra dev
```

The package installs a command-line program named `ldi`, short for Long Document
Indexing. The `ldi run` command reads an experiment configuration and runs the benchmark.

Run all seven systems on the small deterministic fixture:

```bash
uv run --python 3.13 --no-editable \
  --reinstall-package long-document-indexing \
  ldi run --config configs/experiments/advanced-systems-smoke.yaml
```

This run uses local fake models and does not require Azure credentials or paid model
calls. It builds indexes, runs questions through all seven systems, calculates metrics,
and writes reports. Use it to check the software workflow; its fake-model scores do not
measure legal retrieval quality.

Results are written to `artifacts/advanced-systems-smoke/`. Start with
`artifacts/advanced-systems-smoke/report/results.md` for a readable summary.

Run the tests:

```bash
uv run --python 3.13 --extra dev pytest
```

The following sections explain the retrieval approach. To run it against legal case
files with Azure model deployments, continue with
[Run the Legal Benchmark](#run-the-legal-benchmark).

## The Retrieval Problem

A basic RAG system splits documents into passages, or chunks, embeds those passages, and
retrieves the passages most similar to a question:

```text
question -> search every passage -> top passages -> answer model
```

Semantic similarity retrieval scores each chunk independently against the question. It
can miss evidence for multi-hop questions, where a chunk becomes relevant through its
relationship to evidence elsewhere. Long individual documents make this harder by
distributing related information across many chunks and separating each chunk from the
document's broader structure. Multi-document case files add a distinct challenge:
complaints, orders, and settlements play different roles, which chunk-level similarity
does not explicitly represent.

- Evidence may be distributed across several documents.
- Terminology may change between the complaint, later filings, and the final decision.
- A chunk may match the topic but come from the wrong stage of the case.
- Splitting a long document into chunks can obscure relationships between distant
  sections and the document's overall structure.

This repository focuses on the document-level step: constructing maps when documents are
ingested, then using those maps to route each question to the most relevant documents.
After routing, every map-based system uses the same dense retriever to find passages
inside the selected documents. The experiment holds this passage-retrieval step fixed
across systems. Its metrics show whether document routing improves the
context ultimately provided to the answer model.

## What Is a Document Map?

A document map is a document-level representation created when source documents are
ingested. It organizes the document's main contents into structured entries. Each entry
tells the router what the document contains.

Each map contains:

- an overview of the document;
- entries for important facts, events, claims, decisions, or other topics;
- references from every entry back to the source document and source passages.

The core data model is in
[`maps.py`](src/long_document_indexing/domain/maps.py).
A map entry has a `kind`, `label`, `summary`, and `source_references`. Entries may also
have attributes and child entries.

This is a selected entry from an actual map generated by the extended benchmark:

```yaml
document_id: CJ-AL-0007:doc_0001
construction_method: map_reduce
entry:
  id: E1_case
  kind: case_summary
  label: Case overview & parties
  summary: >
    Named plaintiffs (Sharnalle Mitchell; Lorenzo Brown; Courtney Tubbs;
    Tito Williams) sued the City of Montgomery; complaint filed 03/18/2014
    (Case No. 2:14-cv-00186-MEF-CSC) seeking declaratory, injunctive,
    and monetary relief.
  source_references:
    - document_id: CJ-AL-0007:doc_0001
      segment_ids: [CJ-AL-0007:doc_0001:seg_0001]
    - document_id: CJ-AL-0007:doc_0001
      segment_ids: [CJ-AL-0007:doc_0001:seg_0008]
```

Map entries include references to their associated source passages. These references
support quality checks on map construction. The agentic map builder also uses them to
identify source passages that are not yet represented in the map.

## How Map-Based RAG Works in This Repository

For each of the six document-map systems, the repository creates two representations of
the source material: document maps for routing and a dense passage index for evidence
retrieval.

```text
INDEXING

source document -> ordered passages -> map-building method -> document map
       |
       +---------------------------> dense passage index


QUERYING

question
   |
   v
LLM router reads the complete document maps for the case
   |
   v
select up to 3 documents
   |
   v
dense search over raw passages from those documents
   |
   v
retrieve up to 8 raw evidence passages
   |
   v
shared answer model -> answer and citations
```

The router uses the document maps to select documents. The dense retriever then finds
relevant passages in those documents, and the original passages are sent to the answer
model.

The flat-vector baseline does not use document maps or a document router:

```text
question -> dense search over every raw passage -> top 8 passages -> shared answer model
```

Both paths use the same dense retrieval backend and answer model. Therefore, the main
experimental difference is whether document-map routing narrows the search before
passage retrieval.

The shared map query flow is implemented in
[`map_base.py`](src/long_document_indexing/systems/map_base.py).
The baseline is implemented in
[`flat_vector.py`](src/long_document_indexing/systems/flat_vector.py).

## Systems Compared

All six map systems use the query flow above. They differ only in how they construct each
document map. The benchmark tests whether those construction methods preserve different
information and whether that difference affects retrieval.

| System | What it does |
| --- | --- |
| `flat_vector` | Baseline. Searches raw passages across the whole case without maps or document routing. |
| `stuffing` | Sends the whole document to the map builder in one call when it fits the context limit. |
| `map_reduce` | Maps document parts independently, then merges those partial maps. |
| `refine` | Reads passages in order and updates one evolving map after each passage. |
| `hierarchical_map` | Builds small maps, then combines them through several levels. |
| `outline_then_fill` | Creates a document outline first, then fills its sections from the source. |
| `agentic_map` | Uses a bounded inspect-and-revise loop to find and fill missing coverage. |

## Run the Legal Benchmark

Complete the installation in [Quick Start](#quick-start) before following these steps.

The reference legal experiment runs against Azure OpenAI deployments configured through
Microsoft Foundry. Calls to these deployments can incur Azure usage charges. Start with
the two-case smoke configuration before running the 20-case comparison.

### 1. Create the model deployments

The configuration separates five model roles. Environment values must be the names of
deployments that already exist in your Azure resource.

| Role | Environment variable | Model used in the reference run |
| --- | --- | --- |
| Map builder | `FOUNDRY_GENERATOR_DEPLOYMENT` | GPT-5 mini |
| Document router | `FOUNDRY_ROUTER_MODEL` | GPT-5 mini |
| Answer writer | `FOUNDRY_ANSWER_MODEL` | GPT-5 mini |
| Evaluation judge | `FOUNDRY_JUDGE_MODEL` | GPT-5 |
| Passage embeddings | `FOUNDRY_EMBEDDING_MODEL` | text-embedding-3-large |

The role variables may point to the same deployment, but they do not have to. Separate
names make the experiment and its costs easier to inspect.

### 2. Configure the local environment

```bash
cp .env.example .env
```

Fill in `.env` with your endpoint, API key, deployment names, and Foundry project
endpoint. Do not commit `.env`.

Install the optional dependencies:

```bash
uv sync --python 3.13 --extra dev --extra foundry --extra multilexsum
```

API-key inference does not require Azure CLI login. Publishing results to the Foundry
portal uses your Azure identity and may require `az login`.

### 3. Run the two-case smoke benchmark

```bash
uv run --python 3.13 --extra foundry --extra multilexsum \
  --no-editable --reinstall-package long-document-indexing \
  ldi run \
  --config configs/experiments/foundry-multilexsum-legal-rag-qa-role-separated-smoke.yaml
```

Use this run to verify the endpoint, deployments, quotas, structured outputs, embeddings,
and report generation before starting the larger extended run.

### 4. Preview and run the 20-case comparison

First inspect what will be reused, rebuilt, or queried:

```bash
uv run --python 3.13 --extra foundry --extra multilexsum \
  --no-editable --reinstall-package long-document-indexing \
  ldi run \
  --config configs/experiments/foundry-multilexsum-legal-rag-qa-role-separated-extended.yaml \
  --dry-run-budget
```

Then run it:

```bash
uv run --python 3.13 --extra foundry --extra multilexsum \
  --no-editable --reinstall-package long-document-indexing \
  ldi run \
  --config configs/experiments/foundry-multilexsum-legal-rag-qa-role-separated-extended.yaml
```

The run writes checkpoints as it progresses. Repeating the same command reuses valid
indexes and successful answers, and retries missing or failed work. If an existing
artifact no longer matches the current inputs, the command stops before making model
calls to replace it. Review `--dry-run-budget`, then use `--allow-stale-recompute` only
when that rebuild is intentional. `--force` rebuilds everything.

The configured model-call and token limits are safety ceilings. They are not estimates of
the amount the run should consume.

The same command also calculates local metrics and writes reports. Start with the
outputs below; the additional Foundry evaluation steps are optional.

## Where to Find the Output

Each experiment writes to `artifacts/<experiment-id>/`:

```text
artifacts/<experiment-id>/
  indexes/                 built dense indexes and document maps
  runs/                    one normalized answer record per system and question
  evaluations/
    local-metrics.jsonl    deterministic metric records
    foundry/               exported datasets and optional Foundry evaluation results
  report/
    results.md             readable metric tables
    results.csv            metric means
    confidence-intervals.csv
    system-summary.csv     one comparison row per system
    usage-summary.csv      model calls and tokens by system
    summary.json           complete report data
```

Start with `report/results.md` for a readable summary and `report/system-summary.csv` for
cross-system analysis. Inspect `indexes/<system>/maps/` to see the actual generated maps,
and `runs/<system>.jsonl` to trace a question through selected documents, retrieved
passages, answer text, and citations.

The next section explains the main metrics in those reports. For the full output layout
and a code-level account of how the files are produced, see the
[Repository Walkthrough](docs/repo-walkthrough.md).

## What the Benchmark Measures

The extended comparison uses:

- 20 Multi-LexSum legal case files;
- 60 hand-curated questions: 20 single-document, 20 multi-document, and 20 chained
  multi-document questions;
- 7 retrieval systems, producing 420 answer runs.

Each benchmark question is stored as JSON with its gold-standard answer and supporting
evidence. The following excerpt shows the fields used for evaluation:

```json
{
  "query": "Across the complaint and consent decree, what workplace discrimination was alleged and how much did Brown Publishing agree to pay William Hubbard?",
  "ground_truth": {
    "expected_answer": "The complaint alleged a racially hostile work environment at the Xenia facility, and Brown Publishing agreed to pay William Hubbard $24,750 without admitting liability.",
    "relevant_document_ids": [
      "EE-OH-0071:doc_0001",
      "EE-OH-0071:doc_0003"
    ],
    "relevant_segment_ids": [
      "EE-OH-0071:doc_0001:seg_0001",
      "EE-OH-0071:doc_0003:seg_0001"
    ],
    "evidence": [
      {
        "document_id": "EE-OH-0071:doc_0001",
        "segment_id": "EE-OH-0071:doc_0001:seg_0001",
        "quote": "racially hostile work environment at its Xenia, Ohio facility"
      },
      {
        "document_id": "EE-OH-0071:doc_0003",
        "segment_id": "EE-OH-0071:doc_0003:seg_0001",
        "quote": "Without admitting liability, Brown Publishing agrees to pay the sum of $24,750.00 to William Hubbard"
      }
    ]
  }
}
```

During evaluation, each system's selected documents, retrieved passages, answer, and
citations are scored against the corresponding labels above.

The 60 benchmark questions are stored in
[`rag-qa-extended.jsonl`](benchmarks/multilexsum/rag-qa-extended.jsonl), one question per
line. Each question includes its gold-standard answer, required documents and passages,
and supporting quotes.

The benchmark uses a fixed set of 20 Multi-LexSum cases. The
[case manifest](benchmarks/multilexsum/rag-qa-extended-case-manifest.json) lists those
cases and the dataset version they come from, so every run loads the same source
documents.

### Primary Routing Metrics

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

### Downstream Retrieval Metrics

Every document-map system uses the same dense passage retriever. These metrics show how
the routing decision affects the passages that retriever finds. All systems can return
up to eight passages; the local retrieval metrics below inspect the first four.

| Metric | What it measures |
| --- | --- |
| `segment_recall_at_4` | The fraction of required passages found in the first four results. |
| `context_precision_at_4` | Whether relevant passages were ranked ahead of irrelevant passages in the first four results. |
| `context_recall_at_4` | The fraction of required passages found in the first four results. |
| `evidence_quote_recall_at_4` | The fraction of gold evidence quotes found in the first four passages, after normalizing case and whitespace. |

The benchmark also records map validity and completion, citation checks, answer-reference
wording overlap, runtime, and model usage. Complete definitions, including the optional
Foundry evaluators, are in the [Metrics Reference](docs/metrics.md).

## Optional Foundry Evaluation

The main `ldi run` command already computes local metrics and writes the report. The
following commands are optional: one runs additional Foundry evaluators; the other
publishes deterministic benchmark metrics as portal-visible runs. Neither reruns
indexing, retrieval, or answer generation.

### Run Foundry evaluators

Preview the managed evaluation:

```bash
uv run --python 3.13 --extra foundry --extra multilexsum \
  --no-editable --reinstall-package long-document-indexing \
  ldi evaluate-foundry-managed \
  --config configs/experiments/foundry-multilexsum-legal-rag-qa-role-separated-extended.yaml \
  --dry-run
```

Run it:

```bash
uv run --python 3.13 --extra foundry --extra multilexsum \
  --no-editable --reinstall-package long-document-indexing \
  ldi evaluate-foundry-managed \
  --config configs/experiments/foundry-multilexsum-legal-rag-qa-role-separated-extended.yaml
```

The extended config evaluates the same deterministic 20% sample for every system. It
checkpoints each system/evaluator pair, so a retry can reuse completed work.

The configured evaluators include model-based checks for groundedness, relevance,
retrieval quality, and response completeness. The `document_retrieval` evaluator
compares retrieval results with ground-truth relevance labels; it does not use an LLM
judge. See the [Metrics Reference](docs/metrics.md#foundry-evaluators) for the complete
list.

### Publish seven comparable runs to Foundry

Preview the publication plan:

```bash
uv run --python 3.13 --extra foundry --extra multilexsum \
  --no-editable --reinstall-package long-document-indexing \
  ldi publish-foundry-evals \
  --config configs/experiments/foundry-multilexsum-legal-rag-qa-role-separated-extended.yaml \
  --evaluation-name ldi-multilexsum-legal-rag-qa-extended \
  --run-name rag-qa-role-separated-extended-visible \
  --dry-run
```

Remove `--dry-run` to publish. The command creates one 60-row Foundry run for each
system, making the seven systems directly comparable in the Foundry Evaluations table.
Rerunning the same names reuses completed, unchanged systems.

Managed evaluation and portal publication are separate:

- `evaluate-foundry-managed` runs the configured Foundry evaluators and saves their
  results locally.
- `publish-foundry-evals` publishes the deterministic benchmark metrics as visible,
  system-by-system Foundry runs.

## Repository Guide

| Path | What it contains |
| --- | --- |
| `benchmarks/` | Versioned questions, labels, smoke data, and case manifests. |
| `configs/experiments/` | Complete experiment definitions. |
| `configs/systems/` | Settings for each retrieval and map-building method. |
| `prompts/` | Prompts used to build maps, route questions, and answer. |
| `src/long_document_indexing/datasets/` | Dataset loaders and validation. |
| `src/long_document_indexing/systems/` | The seven systems being compared. |
| `src/long_document_indexing/retrieval/` | Dense and local retrieval backends. |
| `src/long_document_indexing/evaluation/` | Local metrics and Foundry integration. |
| `src/long_document_indexing/reporting.py` | Markdown, CSV, and JSON reports. |
| `tests/` | Unit and integration tests. |

For a code-level tour, continue with the
[Repository Walkthrough](docs/repo-walkthrough.md). For complete metric definitions,
read the [Metrics Reference](docs/metrics.md). For details about the legal questions
and dataset licensing, read the
[Multi-LexSum benchmark notes](benchmarks/multilexsum/README.md).

## Scope and Safety

This is an experimental benchmark, not a production legal research system. It is built
to compare retrieval strategies under controlled conditions.

For long paid runs, it provides:

- incremental checkpoints and writes that do not leave half-written files;
- retries for transient model and embedding failures;
- compatibility checks that prevent silent reuse of out-of-date files;
- explicit model-call and token ceilings;
- validation that blocks incomplete runs from being evaluated or published;
- source IDs on map entries, retrieved passages, and citations.

These controls make interrupted experiments recoverable and results inspectable. They do
not make model output legally authoritative.
