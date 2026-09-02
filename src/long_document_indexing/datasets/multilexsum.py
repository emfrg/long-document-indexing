from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from long_document_indexing.domain.benchmark import (
    BenchmarkItem,
    DatasetCapabilities,
    QuestionSet,
)
from long_document_indexing.domain.corpus import Corpus, Document, Segment

from .base import LoadedDataset


class MultiLexSumAdapter:
    """Multi-LexSum corpus adapter.

    Multi-LexSum provides source documents and expert summaries, but not native QA items
    with page-level evidence labels. This adapter therefore requires a separate question
    JSONL file for benchmark evaluation.
    """

    name = "multilexsum"

    def __init__(
        self,
        *,
        question_set_path: Path,
        split: str = "test",
        revision: str | None = None,
        case_manifest: Path | None = None,
        capabilities: DatasetCapabilities | None = None,
    ) -> None:
        self.question_set_path = question_set_path
        self.split = split
        self.revision = revision
        self.case_manifest = case_manifest
        self.capabilities = capabilities or DatasetCapabilities(has_relevant_documents=True)

    def load(self) -> LoadedDataset:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Multi-LexSum loading requires the optional 'multilexsum' extra: "
                "uv sync --extra multilexsum"
            ) from exc

        dataset = load_dataset("allenai/multi_lexsum", split=self.split, revision=self.revision)
        allowed_case_ids = self._load_allowed_case_ids()

        corpora: list[Corpus] = []
        for row in dataset:
            case_id = _case_id(row)
            if allowed_case_ids is not None and case_id not in allowed_case_ids:
                continue
            corpora.append(_row_to_corpus(row=row, case_id=case_id))

        questions = _load_questions(self.question_set_path)
        return LoadedDataset(
            corpora=corpora,
            question_set=QuestionSet(
                id=self.question_set_path.stem,
                items=questions,
                capabilities=self.capabilities,
                metadata={
                    "dataset": "allenai/multi_lexsum",
                    "split": self.split,
                    "revision": self.revision,
                },
            ),
        )

    def _load_allowed_case_ids(self) -> set[str] | None:
        if self.case_manifest is None:
            return None
        payload = json.loads(self.case_manifest.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return {str(item) for item in payload}
        return {str(item) for item in payload.get("case_ids", [])}


def _load_questions(path: Path) -> list[BenchmarkItem]:
    return [
        BenchmarkItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _case_id(row: dict[str, Any]) -> str:
    for key in ("case_id", "id", "summary_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise ValueError("Multi-LexSum row does not expose a recognizable case id")


def _row_to_corpus(*, row: dict[str, Any], case_id: str) -> Corpus:
    sources = row.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"case {case_id!r} has no source document list")

    documents: list[Document] = []
    for index, source_text in enumerate(sources, start=1):
        document_id = f"{case_id}:doc_{index:04d}"
        documents.append(
            Document(
                id=document_id,
                corpus_id=case_id,
                title=f"Source {index}",
                segments=[
                    Segment(
                        id=f"{document_id}:seg_0001",
                        document_id=document_id,
                        order=1,
                        text=str(source_text),
                    )
                ],
                metadata={"source_index": index},
            )
        )

    return Corpus(id=case_id, documents=documents, metadata={"dataset": "multi_lexsum"})
