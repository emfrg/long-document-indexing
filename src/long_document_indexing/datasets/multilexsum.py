from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from long_document_indexing.domain.benchmark import (
    BenchmarkItem,
    DatasetCapabilities,
    GroundTruth,
    QuestionSet,
)
from long_document_indexing.domain.corpus import Corpus, Document, Segment
from long_document_indexing.text import approximate_token_count

from .base import LoadedDataset

_SUMMARY_LENGTHS = ("long", "short", "tiny")
_DEFAULT_CONFIG_NAME = "v20230518"
_DEFAULT_SEGMENT_TOKENS = 1200
_DEFAULT_SEGMENT_OVERLAP_TOKENS = 120


DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]


@dataclass(frozen=True)
class _CaseManifest:
    case_ids: set[str] | None = None
    max_cases: int | None = None
    max_source_documents: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiLexSumAdapter:
    """Multi-LexSum case-file adapter.

    Multi-LexSum is a multi-document legal summarization corpus. It has source
    documents and expert summaries, but no native QA items with page-level gold
    evidence. The adapter can therefore either load explicit BenchmarkItem JSONL
    questions, or generate summary-style BenchmarkItems from a template JSON file.
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
        options: Mapping[str, Any] | None = None,
        dataset_loader: DatasetLoader | None = None,
    ) -> None:
        self.question_set_path = question_set_path
        self.split = split
        self.revision = revision
        self.case_manifest = case_manifest
        self.capabilities = capabilities or DatasetCapabilities(
            has_expected_answers=True,
            has_relevant_documents=True,
            has_reference_summaries=True,
        )
        self.options = dict(options or {})
        self.config_name = _optional_string(
            self.options.get("config_name"),
            field_name="options.config_name",
            default=_DEFAULT_CONFIG_NAME,
        )
        self.trust_remote_code = _bool_option(
            self.options.get("trust_remote_code", True),
            "options.trust_remote_code",
        )
        self.max_cases = _optional_positive_int(self.options.get("max_cases"), "options.max_cases")
        self.max_source_documents = _optional_positive_int(
            self.options.get("max_source_documents"),
            "options.max_source_documents",
        )
        self.segment_tokens = _positive_int(
            self.options.get("segment_tokens", _DEFAULT_SEGMENT_TOKENS),
            "options.segment_tokens",
        )
        self.segment_overlap_tokens = _non_negative_int(
            self.options.get("segment_overlap_tokens", _DEFAULT_SEGMENT_OVERLAP_TOKENS),
            "options.segment_overlap_tokens",
        )
        if self.segment_overlap_tokens >= self.segment_tokens:
            raise ValueError("options.segment_overlap_tokens must be smaller than segment_tokens")
        self.dataset_loader = dataset_loader

    def load(self) -> LoadedDataset:
        dataset = self._load_dataset()
        manifest = self._load_case_manifest()
        allowed_case_ids = manifest.case_ids
        max_cases = manifest.max_cases if manifest.max_cases is not None else self.max_cases
        max_source_documents = (
            manifest.max_source_documents
            if manifest.max_source_documents is not None
            else self.max_source_documents
        )

        corpora: list[Corpus] = []
        rows_by_case_id: dict[str, Mapping[str, Any]] = {}
        for row in dataset:
            case_id = _case_id(row)
            if allowed_case_ids is not None and case_id not in allowed_case_ids:
                continue

            corpora.append(
                _row_to_corpus(
                    row=row,
                    case_id=case_id,
                    segment_tokens=self.segment_tokens,
                    segment_overlap_tokens=self.segment_overlap_tokens,
                    max_source_documents=max_source_documents,
                )
            )
            rows_by_case_id[case_id] = row

            if max_cases is not None and len(corpora) >= max_cases:
                break

        if not corpora:
            raise ValueError("Multi-LexSum adapter loaded no corpora for the requested selection")

        questions, question_metadata = _load_questions(
            self.question_set_path,
            corpora=corpora,
            rows_by_case_id=rows_by_case_id,
        )
        _validate_question_scope(questions, corpora)

        return LoadedDataset(
            corpora=corpora,
            question_set=QuestionSet(
                id=question_metadata.get("id", self.question_set_path.stem),
                items=questions,
                capabilities=self.capabilities,
                metadata={
                    "dataset": "allenai/multi_lexsum",
                    "dataset_config": self.config_name,
                    "split": self.split,
                    "revision": self.revision,
                    "case_manifest": str(self.case_manifest) if self.case_manifest else None,
                    "manifest": manifest.metadata,
                    **question_metadata,
                },
            ),
        )

    def _load_dataset(self) -> Iterable[Mapping[str, Any]]:
        kwargs: dict[str, Any] = {"split": self.split}
        if self.config_name is not None:
            kwargs["name"] = self.config_name
        if self.revision is not None:
            kwargs["revision"] = self.revision
        kwargs["trust_remote_code"] = self.trust_remote_code

        if self.dataset_loader is not None:
            return self.dataset_loader("allenai/multi_lexsum", **kwargs)

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Multi-LexSum loading requires the optional 'multilexsum' extra: "
                "uv sync --extra multilexsum"
            ) from exc

        return load_dataset("allenai/multi_lexsum", **kwargs)

    def _load_case_manifest(self) -> _CaseManifest:
        if self.case_manifest is None:
            return _CaseManifest()

        payload = json.loads(self.case_manifest.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            case_ids = {str(item) for item in payload}
            if not case_ids:
                raise ValueError(f"case manifest has no case ids: {self.case_manifest}")
            return _CaseManifest(case_ids=case_ids)

        if not isinstance(payload, dict):
            raise ValueError(f"case manifest must be a list or object: {self.case_manifest}")

        raw_case_ids = payload.get("case_ids")
        case_ids = None
        if raw_case_ids is not None:
            if not isinstance(raw_case_ids, list):
                raise ValueError("case manifest field 'case_ids' must be a list")
            case_ids = {str(item) for item in raw_case_ids}
            if not case_ids:
                raise ValueError("case manifest field 'case_ids' must not be empty")

        reserved = {"case_ids", "max_cases", "max_source_documents"}
        return _CaseManifest(
            case_ids=case_ids,
            max_cases=_optional_positive_int(payload.get("max_cases"), "case_manifest.max_cases"),
            max_source_documents=_optional_positive_int(
                payload.get("max_source_documents"),
                "case_manifest.max_source_documents",
            ),
            metadata={key: value for key, value in payload.items() if key not in reserved},
        )


def _load_questions(
    path: Path,
    *,
    corpora: list[Corpus],
    rows_by_case_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[BenchmarkItem], dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"question set file is empty: {path}")

    if path.suffix == ".jsonl":
        questions = [
            BenchmarkItem.model_validate_json(line) for line in text.splitlines() if line.strip()
        ]
        return questions, {"id": path.stem, "question_format": "benchmark_jsonl"}

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        questions = [
            BenchmarkItem.model_validate_json(line) for line in text.splitlines() if line.strip()
        ]
        return questions, {"id": path.stem, "question_format": "benchmark_jsonl"}

    if isinstance(payload, dict) and ("templates" in payload or "question_templates" in payload):
        questions = _build_questions_from_templates(
            payload,
            corpora=corpora,
            rows_by_case_id=rows_by_case_id,
        )
        metadata = dict(payload.get("metadata") or {})
        return questions, {
            "id": str(payload.get("id") or path.stem),
            "question_format": "summary_templates",
            **metadata,
        }

    if isinstance(payload, dict) and "items" in payload:
        items = payload["items"]
        if not isinstance(items, list):
            raise ValueError("question set field 'items' must be a list")
        return (
            [BenchmarkItem.model_validate(item) for item in items],
            {"id": str(payload.get("id") or path.stem), "question_format": "benchmark_json"},
        )

    if isinstance(payload, list):
        return (
            [BenchmarkItem.model_validate(item) for item in payload],
            {"id": path.stem, "question_format": "benchmark_json"},
        )

    raise ValueError(f"unsupported question set format: {path}")


def _build_questions_from_templates(
    payload: Mapping[str, Any],
    *,
    corpora: list[Corpus],
    rows_by_case_id: Mapping[str, Mapping[str, Any]],
) -> list[BenchmarkItem]:
    raw_templates = payload.get("templates", payload.get("question_templates"))
    if not isinstance(raw_templates, list) or not raw_templates:
        raise ValueError("question template file must contain a non-empty templates list")

    questions: list[BenchmarkItem] = []
    for corpus in corpora:
        row = rows_by_case_id[corpus.id]
        summaries = _available_summaries(row)
        relevant_document_ids = {document.id for document in corpus.documents}

        for raw_template in raw_templates:
            if not isinstance(raw_template, dict):
                raise ValueError("each question template must be an object")

            summary_length = str(raw_template.get("summary_length", "long")).strip()
            if summary_length not in _SUMMARY_LENGTHS:
                raise ValueError(f"unsupported Multi-LexSum summary length: {summary_length!r}")

            expected_answer = summaries.get(summary_length)
            if expected_answer is None:
                raise ValueError(f"case {corpus.id!r} has no {summary_length!r} expert summary")

            id_suffix = str(raw_template.get("id_suffix") or summary_length).strip()
            if not id_suffix:
                raise ValueError("question template id_suffix must not be blank")

            raw_query = raw_template.get("query")
            if not isinstance(raw_query, str) or not raw_query.strip():
                raise ValueError("question template query must be a non-blank string")

            raw_tags = raw_template.get("tags", [])
            if not isinstance(raw_tags, list):
                raise ValueError("question template tags must be a list")
            tags = {str(tag) for tag in raw_tags if str(tag).strip()}
            tags.update({"multi_lexsum", "case_file", f"{summary_length}_summary"})

            template_metadata = raw_template.get("metadata", {})
            if not isinstance(template_metadata, dict):
                raise ValueError("question template metadata must be an object")

            query = (
                raw_query.replace("{case_id}", corpus.id)
                .replace("{summary_length}", summary_length)
                .strip()
            )
            questions.append(
                BenchmarkItem(
                    id=f"{corpus.id}:{id_suffix}",
                    corpus_id=corpus.id,
                    query=query,
                    ground_truth=GroundTruth(
                        expected_answer=expected_answer,
                        reference_summary=expected_answer,
                        relevant_document_ids=relevant_document_ids,
                    ),
                    tags=tags,
                    metadata={
                        "dataset": "allenai/multi_lexsum",
                        "summary_length": summary_length,
                        "summary_source": "expert",
                        **template_metadata,
                    },
                )
            )

    return questions


def _validate_question_scope(questions: list[BenchmarkItem], corpora: list[Corpus]) -> None:
    corpora_by_id = {corpus.id: corpus for corpus in corpora}
    corpus_ids = set(corpora_by_id)
    unknown_corpus_ids = sorted({question.corpus_id for question in questions} - corpus_ids)
    if unknown_corpus_ids:
        raise ValueError(
            "question set references corpus ids not loaded by the Multi-LexSum adapter: "
            f"{unknown_corpus_ids}"
        )

    for question in questions:
        corpus = corpora_by_id[question.corpus_id]
        documents = corpus.document_by_id()
        segments = corpus.segment_by_id()
        truth = question.ground_truth

        referenced_document_ids = set(truth.relevant_document_ids)
        referenced_document_ids.update(span.document_id for span in truth.evidence)
        unknown_document_ids = sorted(referenced_document_ids - set(documents))
        if unknown_document_ids:
            raise ValueError(
                f"question {question.id!r} references unknown document ids: "
                f"{unknown_document_ids}"
            )

        referenced_segment_ids = set(truth.relevant_segment_ids)
        referenced_segment_ids.update(
            span.segment_id for span in truth.evidence if span.segment_id is not None
        )
        unknown_segment_ids = sorted(referenced_segment_ids - set(segments))
        if unknown_segment_ids:
            raise ValueError(
                f"question {question.id!r} references unknown segment ids: "
                f"{unknown_segment_ids}"
            )

        for span in truth.evidence:
            if span.segment_id is None:
                source_text = "\n".join(
                    segment.text for segment in documents[span.document_id].segments
                )
            else:
                segment = segments[span.segment_id]
                if segment.document_id != span.document_id:
                    raise ValueError(
                        f"question {question.id!r} evidence span {span.segment_id!r} "
                        f"does not belong to document {span.document_id!r}"
                    )
                source_text = segment.text

            if span.quote and _normalized_text(span.quote) not in _normalized_text(source_text):
                raise ValueError(
                    f"question {question.id!r} evidence quote was not found in "
                    f"{span.segment_id or span.document_id!r}"
                )


def _case_id(row: Mapping[str, Any]) -> str:
    for key in ("case_id", "id", "summary_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise ValueError("Multi-LexSum row does not expose a recognizable case id")


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _row_to_corpus(
    *,
    row: Mapping[str, Any],
    case_id: str,
    segment_tokens: int,
    segment_overlap_tokens: int,
    max_source_documents: int | None,
) -> Corpus:
    sources = row.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"case {case_id!r} has no source document list")

    documents: list[Document] = []
    source_count = (
        len(sources) if max_source_documents is None else min(len(sources), max_source_documents)
    )
    for index, raw_source in enumerate(sources[:source_count], start=1):
        source_text, inline_metadata = _source_text_and_metadata(raw_source)
        if not source_text.strip():
            continue

        row_metadata = _row_source_metadata(row, index)
        source_metadata = {**inline_metadata, **row_metadata}
        source_id = _source_identifier(source_metadata, index)
        source_type = _source_type(source_metadata)
        document_id = f"{case_id}:doc_{index:04d}"

        base_segment_metadata = {
            "dataset": "allenai/multi_lexsum",
            "case_id": case_id,
            "source_index": index,
            "source_id": source_id,
            "source_type": source_type,
        }
        segments = _segment_source_text(
            source_text,
            document_id=document_id,
            segment_tokens=segment_tokens,
            segment_overlap_tokens=segment_overlap_tokens,
            metadata=base_segment_metadata,
        )

        documents.append(
            Document(
                id=document_id,
                corpus_id=case_id,
                title=_source_title(source_metadata, index),
                segments=segments,
                metadata={
                    "dataset": "allenai/multi_lexsum",
                    "case_id": case_id,
                    "source_index": index,
                    "source_id": source_id,
                    "source_type": source_type,
                    "source_char_count": len(source_text),
                    "source_approx_token_count": approximate_token_count(source_text),
                    **source_metadata,
                },
            )
        )

    if not documents:
        raise ValueError(f"case {case_id!r} has no non-empty source documents")

    summaries = _available_summaries(row)
    return Corpus(
        id=case_id,
        documents=documents,
        metadata={
            "dataset": "allenai/multi_lexsum",
            "summary_lengths": sorted(summaries),
            "summary_char_counts": {key: len(value) for key, value in summaries.items()},
            "summary_approx_token_counts": {
                key: approximate_token_count(value) for key, value in summaries.items()
            },
            "source_document_count": len(documents),
            "source_segment_count": sum(len(document.segments) for document in documents),
            "case_metadata": _case_metadata(row),
        },
    )


def _segment_source_text(
    text: str,
    *,
    document_id: str,
    segment_tokens: int,
    segment_overlap_tokens: int,
    metadata: Mapping[str, Any],
) -> list[Segment]:
    words = text.split()
    if not words:
        raise ValueError(f"document {document_id!r} has no segmentable text")

    step = segment_tokens - segment_overlap_tokens
    segments: list[Segment] = []
    for segment_index, start in enumerate(range(0, len(words), step), start=1):
        segment_words = words[start : start + segment_tokens]
        if not segment_words:
            continue

        segment_text = " ".join(segment_words)
        segments.append(
            Segment(
                id=f"{document_id}:seg_{segment_index:04d}",
                document_id=document_id,
                order=segment_index,
                text=segment_text,
                metadata={
                    **metadata,
                    "segment_index": segment_index,
                    "source_word_start": start,
                    "source_word_end": start + len(segment_words),
                    "segment_approx_token_count": approximate_token_count(segment_text),
                    "chunking_strategy": "word_window",
                    "segment_tokens": segment_tokens,
                    "segment_overlap_tokens": segment_overlap_tokens,
                },
            )
        )

        if start + segment_tokens >= len(words):
            break

    return segments


def _source_text_and_metadata(raw_source: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(raw_source, Mapping):
        return str(raw_source), {}

    text_key = None
    for candidate in ("text", "source_text", "document_text", "content"):
        if candidate in raw_source:
            text_key = candidate
            break
    if text_key is None:
        raise ValueError("Multi-LexSum source object has no recognizable text field")

    text = str(raw_source[text_key])
    metadata = {str(key): value for key, value in raw_source.items() if key != text_key}
    return text, metadata


def _row_source_metadata(row: Mapping[str, Any], source_index: int) -> dict[str, Any]:
    zero_based_index = source_index - 1
    for key in (
        "source_metadata",
        "sources_metadata",
        "document_metadata",
        "documents_metadata",
        "source_documents",
    ):
        metadata_list = row.get(key)
        if isinstance(metadata_list, list) and zero_based_index < len(metadata_list):
            metadata = metadata_list[zero_based_index]
            if isinstance(metadata, Mapping):
                return {str(item_key): value for item_key, value in metadata.items()}
    return {}


def _source_identifier(metadata: Mapping[str, Any], source_index: int) -> str:
    for key in ("source_id", "document_id", "doc_id", "id", "file_id"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"source_{source_index:04d}"


def _source_type(metadata: Mapping[str, Any]) -> str | None:
    for key in ("source_type", "document_type", "doc_type", "type"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _source_title(metadata: Mapping[str, Any], source_index: int) -> str:
    for key in ("title", "document_title", "name", "file_name", "filename"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    source_type = _source_type(metadata)
    if source_type:
        return f"Source {source_index}: {source_type}"
    return f"Source {source_index}"


def _available_summaries(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        length: summary
        for length in _SUMMARY_LENGTHS
        if (summary := _summary_text(row, length)) is not None
    }


def _summary_text(row: Mapping[str, Any], length: str) -> str | None:
    for key in (
        f"summary/{length}",
        f"{length}_summary",
        f"summary_{length}",
        length,
    ):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)

    for container_key in ("summary", "summaries"):
        container = row.get(container_key)
        if isinstance(container, Mapping):
            value = container.get(length)
            if value is not None and str(value).strip():
                return str(value)

    return None


def _case_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = row.get("case_metadata")
    if isinstance(metadata, Mapping):
        return {str(key): value for key, value in metadata.items()}
    return {}


def _optional_string(value: Any, *, field_name: str, default: str | None) -> str | None:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string when set")
    return value


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _positive_int(value: Any, field_name: str) -> int:
    parsed = _parse_int(value, field_name)
    if parsed < 1:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    parsed = _parse_int(value, field_name)
    if parsed < 0:
        raise ValueError(f"{field_name} must not be negative")
    return parsed


def _parse_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _bool_option(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean")
