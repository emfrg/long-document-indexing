from __future__ import annotations

import json
from pathlib import Path

from long_document_indexing.domain.benchmark import (
    BenchmarkItem,
    DatasetCapabilities,
    QuestionSet,
)
from long_document_indexing.domain.corpus import Corpus

from .base import LoadedDataset


class JsonlDatasetAdapter:
    """Adapter for local corpus JSON plus benchmark-question JSONL files."""

    name = "jsonl"

    def __init__(
        self,
        *,
        corpus_path: Path,
        question_set_path: Path,
        capabilities: DatasetCapabilities | None = None,
        question_set_id: str | None = None,
    ) -> None:
        self.corpus_path = corpus_path
        self.question_set_path = question_set_path
        self.capabilities = capabilities or DatasetCapabilities()
        self.question_set_id = question_set_id or question_set_path.stem

    def load(self) -> LoadedDataset:
        corpora = self._load_corpora()
        questions = self._load_questions()
        return LoadedDataset(
            corpora=corpora,
            question_set=QuestionSet(
                id=self.question_set_id,
                items=questions,
                capabilities=self.capabilities,
                metadata={"source_path": str(self.question_set_path)},
            ),
        )

    def _load_corpora(self) -> list[Corpus]:
        payload = json.loads(self.corpus_path.read_text(encoding="utf-8"))
        if "corpora" in payload:
            return [Corpus.model_validate(item) for item in payload["corpora"]]
        return [Corpus.model_validate(payload)]

    def _load_questions(self) -> list[BenchmarkItem]:
        items: list[BenchmarkItem] = []
        for line_number, line in enumerate(
            self.question_set_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                items.append(BenchmarkItem.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(
                    f"invalid benchmark item at {self.question_set_path}:{line_number}"
                ) from exc
        return items
