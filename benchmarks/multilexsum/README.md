# Multi-LexSum Case-File Benchmark

This benchmark adapter targets `allenai/multi_lexsum`, a multi-document civil-rights
litigation summarization dataset built from Civil Rights Litigation Clearinghouse case
files.

The repo does not commit the raw case files. The adapter loads them through Hugging Face
Datasets into the local machine cache when the `multilexsum` extra is installed.

Licensing notes from the dataset card:

- Dataset package: ODC-By.
- Summaries and metadata: CC BY-NC.
- Source documents: public domain.

The benchmark questions in this directory are template questions. At load time, the
adapter expands each template once per selected case and uses the expert Multi-LexSum
summary as both `expected_answer` and `reference_summary`.

Because these generated questions ask for whole-case summaries, every loaded source
document is treated as relevant. Document-recall metrics are therefore less useful than
summary-reference metrics for this benchmark. The summary smoke configs use local lexical
reference metrics plus citation validity and map-health metrics.

## Curated RAG-QA Overlay

`rag-qa.jsonl` is a small curated QA overlay for RAG-specific evaluation. It uses two
selected case files and labels each question with:

- `expected_answer` for answer-reference checks and Foundry response completeness.
- `relevant_document_ids` for document-routing metrics.
- `relevant_segment_ids` for context precision/recall.
- `evidence` quotes for evidence-retrieval and citation-support checks.

This overlay is intentionally small. Its purpose is to make retrieval, grounding, and
evidence-label plumbing easy to inspect before scaling to a larger hand-labeled legal QA
set.

Run it with:

```bash
uv run --python 3.13 --extra foundry --extra multilexsum --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/foundry-multilexsum-legal-rag-qa-smoke.yaml
```

The corresponding Foundry managed evaluation should use RAG evaluators such as
`groundedness`, `relevance`, `retrieval`, `document_retrieval`, and
`response_completeness`, not just summary-overlap metrics such as ROUGE.

## Extended RAG-QA Comparison

`rag-qa-extended.jsonl` is the fixed comparison set used for the extended study:

- 20 case files across 12 civil-rights legal domains.
- 60 evidence-labelled questions: 20 single-document, 20 multi-document, and 20
  three-stage chained multi-document questions.
- At most eight source documents per case.
- Exact document IDs, segment IDs, and source quotes for deterministic RAG metrics.
- Seven systems, producing 420 system-question runs and exported rows.

The case selection and its provenance are recorded in
`rag-qa-extended-case-manifest.json`. The adapter validates every evidence quote against
the configured 1,200-token source segment before a run begins. The experiment also pins
the exact Hugging Face dataset revision used for annotation.

Use the role-separated extended config for the complete workflow:

```bash
uv run --python 3.13 --extra foundry --extra multilexsum --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/foundry-multilexsum-legal-rag-qa-role-separated-extended.yaml --dry-run-budget
uv run --python 3.13 --extra foundry --extra multilexsum --no-editable --reinstall-package long-document-indexing ldi run --config configs/experiments/foundry-multilexsum-legal-rag-qa-role-separated-extended.yaml
```

The map router can select up to three documents so the chained tier is answerable. The
run is resumable and bounded at 5,000 model calls and 28 million recorded tokens.
Those are safety ceilings, not targets. The preceding two-case role-separated smoke run
is the pilot for deployment and credential validation.
