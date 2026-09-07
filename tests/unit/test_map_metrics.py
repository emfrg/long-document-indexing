from __future__ import annotations

from long_document_indexing.domain.corpus import Corpus, Document, Segment
from long_document_indexing.domain.maps import DocumentMap, IndexArtifact, MapEntry, SourceReference
from long_document_indexing.evaluation.local.maps import (
    evaluate_index_artifact,
    source_reference_validity,
)
from long_document_indexing.storage.artifacts import ArtifactStore
from long_document_indexing.storage.maps import write_document_map


def test_map_metrics_validate_provenance_and_completion(tmp_path) -> None:
    corpus = Corpus(
        id="corpus",
        documents=[
            Document(
                id="doc_alpha",
                corpus_id="corpus",
                segments=[
                    Segment(
                        id="alpha_s1",
                        document_id="doc_alpha",
                        order=1,
                        text="Alpha renewal approval from the board.",
                    )
                ],
            )
        ],
    )
    store = ArtifactStore(tmp_path / "artifacts", "exp")
    document_map = DocumentMap(
        document_id="doc_alpha",
        overview="Alpha renewal approval.",
        entries=[
            MapEntry(
                id="entry_1",
                kind="segment_summary",
                label="Approval",
                summary="Alpha renewal approval from the board.",
                source_references=[
                    SourceReference(document_id="doc_alpha", segment_ids=["alpha_s1"])
                ],
            )
        ],
        construction_method="stuffing",
    )
    map_id, path = write_document_map(
        store,
        system_id="stuffing",
        corpus_id=corpus.id,
        document_map=document_map,
    )
    artifact = IndexArtifact(
        id="index_1",
        system_id="stuffing",
        corpus_id=corpus.id,
        artifact_path="index",
        document_map_ids=[map_id],
        build_metadata={"document_map_paths": {map_id: str(path)}},
    )

    metrics = evaluate_index_artifact(
        artifact,
        corpus,
        store,
        [
            "map_schema_validity",
            "map_source_reference_validity",
            "map_compression_ratio",
            "map_completion_rate",
        ],
        experiment_id="exp",
    )
    by_name = {metric.name: metric.value for metric in metrics}

    assert by_name["map_schema_validity"] == 1.0
    assert by_name["map_source_reference_validity"] == 1.0
    assert by_name["map_completion_rate"] == 1.0
    assert by_name["map_compression_ratio"] > 0.0


def test_source_reference_validity_rejects_empty_segment_references() -> None:
    corpus = _single_document_corpus()
    document_map = DocumentMap(
        document_id="doc_alpha",
        overview="Alpha map.",
        entries=[
            MapEntry(
                id="entry_1",
                kind="segment_summary",
                label="Approval",
                summary="Alpha renewal approval from the board.",
                source_references=[SourceReference(document_id="doc_alpha", segment_ids=[])],
            )
        ],
        construction_method="stuffing",
    )

    assert source_reference_validity([document_map], corpus) == 0.0


def _single_document_corpus() -> Corpus:
    return Corpus(
        id="corpus",
        documents=[
            Document(
                id="doc_alpha",
                corpus_id="corpus",
                segments=[
                    Segment(
                        id="alpha_s1",
                        document_id="doc_alpha",
                        order=1,
                        text="Alpha renewal approval from the board.",
                    )
                ],
            )
        ],
    )
