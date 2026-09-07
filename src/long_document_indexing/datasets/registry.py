from __future__ import annotations

from long_document_indexing.config import DatasetConfig

from .base import DatasetAdapter
from .jsonl import JsonlDatasetAdapter
from .multilexsum import MultiLexSumAdapter


def create_dataset_adapter(config: DatasetConfig) -> DatasetAdapter:
    if config.adapter == "jsonl":
        if config.corpus_path is None or config.question_set is None:
            raise ValueError("jsonl dataset adapter requires corpus_path and question_set")
        return JsonlDatasetAdapter(
            corpus_path=config.corpus_path,
            question_set_path=config.question_set,
            capabilities=config.capabilities,
        )

    if config.adapter == "multilexsum":
        if config.question_set is None:
            raise ValueError("multilexsum dataset adapter requires question_set")
        return MultiLexSumAdapter(
            question_set_path=config.question_set,
            split=config.split or "test",
            revision=config.revision,
            case_manifest=config.case_manifest,
            capabilities=config.capabilities,
            options=config.options,
        )

    raise ValueError(f"unknown dataset adapter: {config.adapter}")
