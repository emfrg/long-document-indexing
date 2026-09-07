from __future__ import annotations

import json
from pathlib import Path

import pytest

from long_document_indexing.config import FoundryEvaluationConfig
from long_document_indexing.domain.benchmark import BenchmarkItem, EvidenceSpan, GroundTruth
from long_document_indexing.domain.runs import Citation, RagRunRecord, RetrievedItem, UsageRecord
from long_document_indexing.evaluation.foundry import (
    build_foundry_evaluation_rows,
    write_foundry_evaluation_export,
)
from long_document_indexing.storage.artifacts import ArtifactStore


def test_foundry_evaluation_rows_preserve_query_response_context_and_truth() -> None:
    rows = build_foundry_evaluation_rows([_record()], {"q_alpha": _item()})

    assert len(rows) == 1
    row = rows[0]
    assert row.id == "run-alpha"
    assert row.query == "What did the board approve?"
    assert row.response == "The board approved the Alpha renewal."
    assert row.ground_truth == "The board approved the Alpha renewal."
    assert "[rank=1 document_id=doc_alpha segment_id=alpha_s1]" in row.context
    assert row.messages[0].role == "user"
    assert row.messages[1].role == "assistant"
    assert [item.segment_id for item in row.retrieved_context] == ["alpha_s1", "alpha_s2"]
    assert [item.document_id for item in row.retrieved_documents] == ["doc_alpha"]
    assert row.retrieval_ground_truth[0].query_relevance_label == 4
    assert row.relevant_document_ids == ["doc_alpha"]
    assert row.relevant_segment_ids == ["alpha_s1"]
    assert row.expected_behavior.evidence[0].quote == "approval"
    assert row.metadata["usage"]["model_calls"] == 1


def test_foundry_evaluation_rows_reject_unknown_benchmark_items() -> None:
    with pytest.raises(KeyError, match="unknown benchmark item"):
        build_foundry_evaluation_rows([_record()], {})


def test_write_foundry_evaluation_export_writes_dataset_and_manifest(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts", "exp")

    export = write_foundry_evaluation_export(
        store=store,
        records=[_record()],
        items_by_id={"q_alpha": _item()},
        config=FoundryEvaluationConfig(
            enabled=True,
            dataset_path=Path("eval/foundry.jsonl"),
            manifest_path=Path("eval/foundry-manifest.json"),
        ),
        experiment_id="exp",
    )

    assert export.row_count == 1
    dataset_rows = [
        json.loads(line)
        for line in export.dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(export.manifest_path.read_text(encoding="utf-8"))

    assert dataset_rows[0]["query"] == "What did the board approve?"
    assert dataset_rows[0]["context"].startswith("[rank=1 document_id=doc_alpha")
    assert manifest["dataset_path"] == "eval/foundry.jsonl"
    assert manifest["row_count"] == 1
    assert manifest["evaluation_level"] == "turn"
    assert manifest["foundry_data_mapping"]["query"] == "{{item.query}}"
    assert manifest["evaluator_data_mappings"]["document_retrieval"] == {
        "retrieval_ground_truth": "{{item.retrieval_ground_truth}}",
        "retrieved_documents": "{{item.retrieved_documents}}",
    }
    assert manifest["azure_ai_evaluation_column_mapping"]["response"] == "${data.response}"


def _item() -> BenchmarkItem:
    return BenchmarkItem(
        id="q_alpha",
        corpus_id="smoke-corpus",
        query="What did the board approve?",
        ground_truth=GroundTruth(
            expected_answer="The board approved the Alpha renewal.",
            relevant_document_ids={"doc_alpha"},
            relevant_segment_ids={"alpha_s1"},
            evidence=[
                EvidenceSpan(
                    document_id="doc_alpha",
                    segment_id="alpha_s1",
                    start_char=10,
                    end_char=18,
                    quote="approval",
                )
            ],
        ),
        tags={"smoke", "approval"},
        metadata={"source": "unit"},
    )


def _record() -> RagRunRecord:
    return RagRunRecord(
        run_id="run-alpha",
        experiment_id="exp",
        system_id="stuffing",
        corpus_id="smoke-corpus",
        item_id="q_alpha",
        selected_document_ids=["doc_alpha"],
        retrieved_items=[
            RetrievedItem(
                document_id="doc_alpha",
                segment_id="alpha_s2",
                text="Finance estimated a cost reduction.",
                score=0.75,
                rank=2,
                retrieval_stage="map",
            ),
            RetrievedItem(
                document_id="doc_alpha",
                segment_id="alpha_s1",
                text="The board approved the Alpha renewal.",
                score=0.98,
                rank=1,
                retrieval_stage="map",
            ),
        ],
        answer="The board approved the Alpha renewal.",
        citations=[
            Citation(
                document_id="doc_alpha",
                segment_id="alpha_s1",
                quote="The board approved the Alpha renewal.",
            )
        ],
        usage=UsageRecord(input_tokens=10, output_tokens=12, model_calls=1, tool_calls=1),
        trace_id="trace-alpha",
        workflow_artifact_path="workflows/query/stuffing/q_alpha/rep-0.json",
        status="succeeded",
    )
