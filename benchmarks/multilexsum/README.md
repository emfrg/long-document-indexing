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
