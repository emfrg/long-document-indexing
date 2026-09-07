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

