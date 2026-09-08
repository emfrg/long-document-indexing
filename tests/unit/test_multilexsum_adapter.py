from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from long_document_indexing.config import load_experiment_config
from long_document_indexing.datasets.multilexsum import MultiLexSumAdapter
from long_document_indexing.datasets.registry import create_dataset_adapter


def test_multilexsum_adapter_segments_selected_cases_and_generates_questions(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "case_ids": ["case-alpha"],
                "max_source_documents": 2,
                "note": "unit-test selection",
            }
        ),
        encoding="utf-8",
    )
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            {
                "id": "unit-multilexsum",
                "templates": [
                    {
                        "id_suffix": "long_summary",
                        "summary_length": "long",
                        "query": "Summarize case {case_id} using the {summary_length} form.",
                        "tags": ["legal"],
                        "metadata": {"answer_style": "long"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, dict[str, Any]]] = []

    def dataset_loader(path: str, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append((path, kwargs))
        return [
            {
                "id": "case-alpha",
                "sources": [
                    {
                        "text": _words("complaint", 45),
                        "id": "complaint-source",
                        "document_type": "complaint",
                        "title": "Complaint",
                    },
                    _words("order", 18),
                ],
                "source_metadata": [
                    {"source_id": "complaint-source", "type": "complaint"},
                    {"source_id": "order-source", "type": "order", "title": "Order"},
                ],
                "summary/long": "Alpha expert long summary.",
                "summary/short": "Alpha short summary.",
                "case_metadata": {"case_name": "Alpha v. City"},
            },
            {
                "id": "case-beta",
                "sources": [_words("beta", 30)],
                "summary/long": "Beta expert long summary.",
            },
        ]

    adapter = MultiLexSumAdapter(
        question_set_path=questions_path,
        case_manifest=manifest_path,
        options={"segment_tokens": 20, "segment_overlap_tokens": 5},
        dataset_loader=dataset_loader,
    )

    loaded = adapter.load()

    assert calls == [
        (
            "allenai/multi_lexsum",
            {"split": "test", "name": "v20230518", "trust_remote_code": True},
        )
    ]
    assert [corpus.id for corpus in loaded.corpora] == ["case-alpha"]

    corpus = loaded.corpora[0]
    assert corpus.metadata["summary_lengths"] == ["long", "short"]
    assert corpus.metadata["source_document_count"] == 2
    assert corpus.metadata["source_segment_count"] == 4
    assert corpus.metadata["case_metadata"] == {"case_name": "Alpha v. City"}

    complaint = corpus.documents[0]
    assert complaint.id == "case-alpha:doc_0001"
    assert complaint.title == "Complaint"
    assert complaint.metadata["source_id"] == "complaint-source"
    assert complaint.metadata["source_type"] == "complaint"
    assert [segment.order for segment in complaint.segments] == [1, 2, 3]
    assert [segment.id for segment in complaint.segments] == [
        "case-alpha:doc_0001:seg_0001",
        "case-alpha:doc_0001:seg_0002",
        "case-alpha:doc_0001:seg_0003",
    ]
    assert complaint.segments[1].metadata["source_word_start"] == 15
    assert complaint.segments[1].metadata["segment_overlap_tokens"] == 5

    question_set = loaded.question_set
    assert question_set.id == "unit-multilexsum"
    assert question_set.capabilities.has_expected_answers is True
    assert question_set.capabilities.has_reference_summaries is True
    assert question_set.metadata["question_format"] == "summary_templates"

    question = question_set.items[0]
    assert question.id == "case-alpha:long_summary"
    assert question.corpus_id == "case-alpha"
    assert question.query == "Summarize case case-alpha using the long form."
    assert question.ground_truth.expected_answer == "Alpha expert long summary."
    assert question.ground_truth.reference_summary == "Alpha expert long summary."
    assert question.ground_truth.relevant_document_ids == {
        "case-alpha:doc_0001",
        "case-alpha:doc_0002",
    }
    assert question.ground_truth.relevant_segment_ids == set()
    assert {"multi_lexsum", "case_file", "long_summary", "legal"} <= question.tags
    assert question.metadata["answer_style"] == "long"


def test_multilexsum_manifest_can_select_first_cases(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"max_cases": 1, "max_source_documents": 1}),
        encoding="utf-8",
    )
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            {
                "templates": [
                    {
                        "id_suffix": "summary",
                        "summary_length": "long",
                        "query": "Summarize the case file.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    adapter = MultiLexSumAdapter(
        question_set_path=questions_path,
        case_manifest=manifest_path,
        options={"segment_tokens": 50, "segment_overlap_tokens": 0},
        dataset_loader=lambda *_args, **_kwargs: [
            {"id": "case-alpha", "sources": [_words("alpha", 10)], "summary/long": "Alpha."},
            {"id": "case-beta", "sources": [_words("beta", 10)], "summary/long": "Beta."},
        ],
    )

    loaded = adapter.load()

    assert [corpus.id for corpus in loaded.corpora] == ["case-alpha"]
    assert len(loaded.corpora[0].documents) == 1
    assert [item.corpus_id for item in loaded.question_set.items] == ["case-alpha"]


def test_multilexsum_casefile_smoke_config_is_benchmark_ready() -> None:
    config = load_experiment_config(
        Path("configs/experiments/multilexsum-casefile-smoke.yaml"),
        project_root=Path.cwd(),
    )
    adapter = create_dataset_adapter(config.dataset)

    assert isinstance(adapter, MultiLexSumAdapter)
    assert config.dataset.question_set == Path.cwd() / "benchmarks/multilexsum/questions.json"
    assert config.dataset.case_manifest == Path.cwd() / "benchmarks/multilexsum/case-manifest.json"
    assert config.dataset.options["config_name"] == "v20230518"
    assert config.dataset.options["trust_remote_code"] is True
    assert adapter.segment_tokens == 1200
    assert adapter.segment_overlap_tokens == 120
    assert adapter.trust_remote_code is True
    assert config.answering.mode == "generated"
    assert config.evaluation.local == [
        "map_schema_validity",
        "map_source_reference_validity",
        "map_compression_ratio",
        "map_completion_rate",
        "answer_reference_similarity",
        "answer_reference_token_precision",
        "answer_reference_token_recall",
        "answer_reference_token_f1",
        "invalid_citation_rate",
        "query_duration_ms",
        "tool_calls",
    ]
    assert config.evaluation.foundry.evaluators == ["groundedness", "relevance"]
    assert config.evaluation.foundry.managed_evaluators == ["f1", "rouge"]
    assert config.systems == [
        "flat_vector",
        "stuffing",
        "map_reduce",
        "refine",
        "hierarchical_map",
        "outline_then_fill",
        "agentic_map",
    ]
    assert config.evaluation.foundry.enabled is True


def test_foundry_multilexsum_legal_thin_slice_config_is_bounded_real_run() -> None:
    config = load_experiment_config(
        Path("configs/experiments/foundry-multilexsum-legal-thin-slice.yaml"),
        project_root=Path.cwd(),
    )

    assert config.models.generator_provider == "foundry"
    assert config.models.generator_api == "responses"
    assert config.models.generator_response_format == "structured"
    assert config.answering.mode == "generated"
    assert config.systems == [
        "flat_vector",
        "stuffing",
        "map_reduce",
        "refine",
        "hierarchical_map",
        "outline_then_fill",
        "agentic_map",
    ]
    assert config.run_control.resume is True
    assert config.run_control.max_model_calls == 1400
    assert config.run_control.max_total_tokens == 7_000_000
    assert config.dataset.adapter == "multilexsum"
    assert config.dataset.options["trust_remote_code"] is True
    assert config.evaluation.local == [
        "map_schema_validity",
        "map_source_reference_validity",
        "map_compression_ratio",
        "map_completion_rate",
        "answer_reference_similarity",
        "answer_reference_token_precision",
        "answer_reference_token_recall",
        "answer_reference_token_f1",
        "invalid_citation_rate",
        "query_duration_ms",
        "tool_calls",
    ]
    assert config.evaluation.foundry.evaluators == ["groundedness", "relevance"]


def test_foundry_multilexsum_clean_all_systems_config_is_bounded_real_run() -> None:
    config = load_experiment_config(
        Path("configs/experiments/foundry-multilexsum-legal-clean-all-systems.yaml"),
        project_root=Path.cwd(),
    )

    assert config.experiment.id == "foundry-multilexsum-legal-clean-all-systems"
    assert config.models.generator_provider == "foundry"
    assert config.models.generator_api == "responses"
    assert config.models.generator_response_format == "structured"
    assert config.answering.mode == "generated"
    assert config.systems == [
        "flat_vector",
        "stuffing",
        "map_reduce",
        "refine",
        "hierarchical_map",
        "outline_then_fill",
        "agentic_map",
    ]
    assert config.run_control.resume is True
    assert config.run_control.max_model_calls == 1400
    assert config.run_control.max_total_tokens == 7_000_000
    assert config.dataset.adapter == "multilexsum"
    assert config.dataset.options["trust_remote_code"] is True
    assert config.evaluation.foundry.enabled is True
    assert config.evaluation.foundry.evaluators == ["groundedness", "relevance"]


def _words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}_{index}" for index in range(count))
