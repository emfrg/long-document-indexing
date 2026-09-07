from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from long_document_indexing.domain.corpus import Corpus
from long_document_indexing.domain.maps import DocumentMap, IndexArtifact
from long_document_indexing.domain.runs import MetricRecord
from long_document_indexing.storage.artifacts import ArtifactStore
from long_document_indexing.text import approximate_token_count

INDEX_ITEM_ID = "__index__"


def evaluate_index_artifact(
    artifact: IndexArtifact,
    corpus: Corpus,
    store: ArtifactStore,
    metric_names: Iterable[str],
    *,
    experiment_id: str,
) -> list[MetricRecord]:
    del store
    if "document_map_paths" not in artifact.build_metadata:
        return []

    map_metric_names = [name for name in metric_names if name.startswith("map_")]
    if not map_metric_names:
        return []

    document_maps = _load_document_maps(artifact)
    return [
        MetricRecord(
            experiment_id=experiment_id,
            run_id=artifact.id,
            system_id=artifact.system_id,
            corpus_id=artifact.corpus_id,
            item_id=INDEX_ITEM_ID,
            level="document_map",
            name=metric_name,
            value=_calculate_map_metric(metric_name, document_maps, artifact, corpus),
            metadata={"artifact_id": artifact.id},
        )
        for metric_name in map_metric_names
    ]


def source_reference_validity(document_maps: list[DocumentMap], corpus: Corpus) -> float:
    document_ids = set(corpus.document_by_id())
    segments = corpus.segment_by_id()

    total_references = 0
    valid_references = 0
    for document_map in document_maps:
        for root in document_map.entries:
            for entry in root.walk():
                for reference in entry.source_references:
                    total_references += 1
                    if reference.document_id not in document_ids:
                        continue
                    if reference.document_id != document_map.document_id:
                        continue
                    if not reference.segment_ids:
                        continue
                    if all(
                        segment_id in segments
                        and segments[segment_id].document_id == reference.document_id
                        for segment_id in reference.segment_ids
                    ):
                        valid_references += 1

    if total_references == 0:
        return 0.0
    return valid_references / total_references


def compression_ratio(document_maps: list[DocumentMap], corpus: Corpus) -> float:
    source_tokens = sum(
        approximate_token_count(segment.text)
        for document in corpus.documents
        for segment in document.segments
    )
    map_tokens = sum(
        approximate_token_count(_map_text(document_map)) for document_map in document_maps
    )
    if source_tokens == 0:
        return 0.0
    return map_tokens / source_tokens


def _calculate_map_metric(
    metric_name: str,
    document_maps: list[DocumentMap],
    artifact: IndexArtifact,
    corpus: Corpus,
) -> float:
    if metric_name == "map_schema_validity":
        paths = _document_map_paths(artifact)
        if not paths:
            return 0.0
        return len(document_maps) / len(paths)

    if metric_name == "map_source_reference_validity":
        return source_reference_validity(document_maps, corpus)

    if metric_name == "map_compression_ratio":
        return compression_ratio(document_maps, corpus)

    if metric_name == "map_completion_rate":
        return len(document_maps) / len(corpus.documents)

    raise ValueError(f"unknown map metric: {metric_name}")


def _load_document_maps(artifact: IndexArtifact) -> list[DocumentMap]:
    document_maps = []
    for path in _document_map_paths(artifact).values():
        try:
            document_maps.append(
                DocumentMap.model_validate_json(Path(path).read_text(encoding="utf-8"))
            )
        except ValueError:
            continue
    return document_maps


def _document_map_paths(artifact: IndexArtifact) -> dict[str, str]:
    paths = artifact.build_metadata.get("document_map_paths", {})
    if not isinstance(paths, dict):
        return {}
    return {str(key): str(value) for key, value in paths.items()}


def _map_text(document_map: DocumentMap) -> str:
    parts = [document_map.overview, str(document_map.facets)]
    for root in document_map.entries:
        for entry in root.walk():
            parts.extend([entry.kind, entry.label, entry.summary, str(entry.attributes)])
    return "\n".join(parts)
