from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from long_document_indexing.domain.benchmark import QuestionSet
from long_document_indexing.domain.corpus import Corpus


@dataclass(frozen=True)
class LoadedDataset:
    """Corpus plus evaluator-only question set loaded from one adapter."""

    corpora: list[Corpus]
    question_set: QuestionSet


class DatasetAdapter(Protocol):
    name: str

    def load(self) -> LoadedDataset:
        """Load canonical corpora and benchmark questions."""
