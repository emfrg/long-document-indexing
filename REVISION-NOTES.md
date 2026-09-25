# Documentation Revision Notes

## Source

These drafts are based on `emfrg/long-document-indexing` at commit
`e140d1154c301dae4de2806e71da7930dc67befa`:

- [Original README](https://github.com/emfrg/long-document-indexing/blob/e140d1154c301dae4de2806e71da7930dc67befa/README.md)
- [Original repository walkthrough](https://github.com/emfrg/long-document-indexing/blob/e140d1154c301dae4de2806e71da7930dc67befa/docs/repo-walkthrough.md)

The drafts incorporate the previously reviewed documentation corrections. They do not
change the implementation, commands, model-role settings, dataset, or reported results.
No changes have been made to the GitHub repository.

## Files

```text
README.md
docs/
  repo-walkthrough.md
  metrics.md
REVISION-NOTES.md
```

`README.md` and `docs/repo-walkthrough.md` replace the corresponding documentation
files. `docs/metrics.md` is a new reference containing the relocated metric catalogue.
The Markdown links assume these paths within the existing repository. This package
contains documentation only, not the source code or benchmark assets linked from it.

This notes file is an editorial handover, not a required part of the repository.

## README

### Moved

- Promoted the existing repository-purpose paragraph above the legal motivation.
- Moved “Try It Locally” near the beginning and renamed it “Quick Start.”
- Placed the real benchmark instructions after the explanation of the compared systems.
- Placed output inspection immediately after the benchmark instructions.
- Moved the detailed metric catalogue to `docs/metrics.md`. The benchmark description,
  question example, primary routing table, and downstream retrieval table remain in
  the README.

### Added

- A short navigation line below the introduction.
- An instruction to run commands from the repository root.
- The specific report to open after the local smoke run.
- A distinction between checking the local software workflow and measuring legal
  retrieval quality.
- Short transitions from the local example to the real benchmark, and from execution
  to report inspection and metric interpretation.
- Links to the complete metric reference.

### Corrected or clarified

- Introduced the RAG abbreviation in the opening paragraph and removed the promoted
  paragraph's dependency on an earlier mention of distributed evidence.
- Specified that the router reads the maps for the current case.
- Described evaluation of selected documents and retrieved passages as well as answers
  and citations.
- Explained how the flat-vector baseline receives document-level scores.
- Distinguished the eight-passage retrieval budget from the four-passage local metrics.
- Clarified passage-label recall and quote matching.
- Renamed the Foundry section to “Optional Foundry Evaluation.”
- Distinguished model-based Foundry evaluators from label-based `document_retrieval`.
- Preserved the distinction between additional evaluator runs and portal publication.

## Repository Walkthrough

### Preserved

- The ten numbered sections and their order.
- The corpus/document/segment definitions.
- The indexing and query diagrams, strategy descriptions, run-record table, output
  layout, command examples, and code-location guide.

### Added or clarified

- Named `ldi run` rather than the bare `ldi` command when describing the full lifecycle.
- Connected configuration to dataset loading, loaded objects to their later use,
  services to indexing, index artifacts to querying, and query results to run records.
- Explained the baseline's document-level scores using the same wording as the README.
- Described resume-enabled behavior and the separate `--force` option consistently with
  the README.
- Stated that `ldi run` already evaluates and reports, so the individual commands are
  for working with those stages separately.
- Described lexical answer comparisons without treating them as semantic completeness.
- Scoped the 420-row export to the complete extended experiment.
- Identified the Foundry evaluator and publication commands as optional.
- Updated the links to Quick Start, Optional Foundry Evaluation, and the metric reference.
- Made a few local grammar and clarity edits without replacing the explanatory sections.

## Metrics Reference

The reference retains the metric names and core descriptions from the original README.
It adds the small interpretation notes established in the preceding review:

- Required-document coverage is binary for each question.
- Context precision is rank-aware; the two passage-recall metrics coincide on this
  benchmark's passage labels.
- Quote-based checks normalize case and whitespace.
- Map completion does not mean complete coverage of the source, and source-reference
  validity does not verify the factual accuracy of a map entry.
- Map compression is a size ratio and can exceed 1.
- Citation quote matching and answer-wording overlap are not semantic correctness tests.
- `tool_calls` is distinct from model-call accounting.
- `document_retrieval` is not an LLM-judge metric.

## Review Scope

The drafts use the GitHub source documents and the previously reviewed code-level
clarifications. The benchmark has not been rerun, and no fresh installation or Azure
execution has been performed. No article link has been invented; the published article
can be linked once its final URL is available.

The documentation was parsed as Markdown, links between the included documents and
their heading anchors were checked, the JSON and YAML examples were parsed, and the
shell examples were syntax-checked without executing their commands. Source-code and
benchmark links retain their repository-relative paths.
