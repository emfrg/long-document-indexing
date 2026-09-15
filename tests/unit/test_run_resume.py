from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from long_document_indexing.cli import (
    _index,
    _is_reusable_index_artifact,
    _is_reusable_query_record,
    _load_index_artifacts_if_present,
    _merge_usage_events,
    _prepare,
    _query,
)
from long_document_indexing.config import RunControlConfig, load_experiment_config
from long_document_indexing.domain.maps import (
    DocumentMap,
    IndexArtifact,
    MapEntry,
    SourceReference,
)
from long_document_indexing.domain.runs import RagRunRecord
from long_document_indexing.prompt_safety import PROMPT_SAFETY_POLICY_VERSION
from long_document_indexing.run_control import BudgetExceeded, BudgetLedger
from long_document_indexing.storage.artifacts import ArtifactStore
from long_document_indexing.storage.maps import (
    DOCUMENT_MAP_CONTENT_SIGNATURE_POLICY_VERSION,
    document_map_content_signature,
    read_document_map,
    write_document_map,
)
from long_document_indexing.systems.map_base import (
    MAP_SOURCE_REFERENCE_NORMALIZATION_POLICY_VERSION,
)
from long_document_indexing.telemetry.usage import UsageEvent
from long_document_indexing.workflows.common_query import (
    QUERY_RUN_POLICY_VERSION,
    index_artifact_signature,
)


def test_resume_reuses_succeeded_index_and_query_artifacts(tmp_path, capsys) -> None:
    config = _foundry_export_smoke_config(tmp_path)

    _prepare(config)
    asyncio.run(_index(config))
    asyncio.run(_query(config))

    experiment_dir = tmp_path / "artifacts" / "foundry-eval-export-smoke"
    index_usage_count = _jsonl_count(experiment_dir / "costs/index-usage.jsonl")
    query_usage_count = _jsonl_count(experiment_dir / "costs/query-usage.jsonl")
    run_count = _jsonl_count(experiment_dir / "runs/stuffing.jsonl")
    initial_output = capsys.readouterr().out
    assert "Query results for stuffing: succeeded=2, failed=0" in initial_output

    resume_config = config.model_copy(update={"run_control": RunControlConfig(resume=True)})
    asyncio.run(_index(resume_config))
    asyncio.run(_query(resume_config))

    records = _run_records(experiment_dir / "runs/stuffing.jsonl")
    assert _jsonl_count(experiment_dir / "costs/index-usage.jsonl") == index_usage_count
    assert _jsonl_count(experiment_dir / "costs/query-usage.jsonl") == query_usage_count
    assert _jsonl_count(experiment_dir / "runs/stuffing.jsonl") == run_count
    assert len(records) == 2
    assert {record.status for record in records} == {"succeeded"}
    assert {record.query_policy_version for record in records} == {QUERY_RUN_POLICY_VERSION}
    assert all(record.index_artifact_id for record in records)
    assert all(record.index_artifact_signature for record in records)
    assert all(record.query_config_signature for record in records)
    output = capsys.readouterr().out
    assert "Indexing system 1/1: stuffing" in output
    assert "stuffing reused case 1/1: smoke-corpus" in output
    assert "Querying system 1/1: stuffing" in output
    assert "stuffing reused question 1/2: q_alpha_approval" in output
    assert "Query results for stuffing: succeeded=0, failed=0, reused=2" in output


def test_stale_map_artifacts_are_not_reusable(tmp_path) -> None:
    artifact_path = tmp_path / "index.json"
    map_path = tmp_path / "map.json"
    artifact_path.write_text("{}", encoding="utf-8")
    map_path.write_text("{}", encoding="utf-8")

    current = _index_artifact(
        artifact_path=artifact_path,
        map_path=map_path,
        prompt_safety_policy=PROMPT_SAFETY_POLICY_VERSION,
        source_reference_normalization_policy=(MAP_SOURCE_REFERENCE_NORMALIZATION_POLICY_VERSION),
    )
    stale_prompt = _index_artifact(
        artifact_path=artifact_path,
        map_path=map_path,
        prompt_safety_policy="old-prompt-policy",
        source_reference_normalization_policy=(MAP_SOURCE_REFERENCE_NORMALIZATION_POLICY_VERSION),
    )
    stale_normalization = _index_artifact(
        artifact_path=artifact_path,
        map_path=map_path,
        prompt_safety_policy=PROMPT_SAFETY_POLICY_VERSION,
        source_reference_normalization_policy="old-normalization-policy",
    )

    assert _is_reusable_index_artifact(current) is True
    assert _is_reusable_index_artifact(stale_prompt) is False
    assert _is_reusable_index_artifact(stale_normalization) is False


def test_document_map_content_signature_ignores_json_key_order() -> None:
    document_map = _document_map()
    payload = document_map.model_dump(mode="json")
    payload["facets"] = {"alpha": 2, "zeta": 1}
    payload["entries"][0]["attributes"] = {"alpha": 2, "zeta": 1}
    reordered = DocumentMap.model_validate(payload)

    assert document_map_content_signature(document_map) == document_map_content_signature(reordered)


def test_current_map_content_signature_detects_changed_map(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts", "experiment")
    document_map = _document_map()
    map_id, map_path = write_document_map(
        store,
        system_id="stuffing",
        corpus_id="corpus",
        document_map=document_map,
    )
    artifact_path = store.path("indexes/stuffing/index-current.json")
    artifact_path.write_text("{}", encoding="utf-8")
    artifact = IndexArtifact(
        id="index-current",
        system_id="stuffing",
        corpus_id="corpus",
        artifact_path=str(artifact_path),
        document_map_ids=[map_id],
        build_metadata={
            "retrieval_backend": "DenseVectorBackend",
            "index_config_signature": "index-config-current",
            "document_map_paths": {map_id: str(map_path)},
            "document_map_signatures": {map_id: document_map_content_signature(document_map)},
            "document_map_signature_policy": (DOCUMENT_MAP_CONTENT_SIGNATURE_POLICY_VERSION),
            "prompt_safety_policy": PROMPT_SAFETY_POLICY_VERSION,
            "source_reference_normalization_policy": (
                MAP_SOURCE_REFERENCE_NORMALIZATION_POLICY_VERSION
            ),
        },
    )

    assert _is_reusable_index_artifact(
        artifact,
        expected_retrieval_backend="DenseVectorBackend",
        expected_index_config_signature="index-config-current",
    )

    changed = read_document_map(map_path).model_copy(update={"overview": "Changed overview"})
    store.write_json(map_path.relative_to(store.experiment_dir), changed)
    assert not _is_reusable_index_artifact(
        artifact,
        expected_retrieval_backend="DenseVectorBackend",
        expected_index_config_signature="index-config-current",
    )


def test_index_discovery_recovers_individual_artifact_without_manifest(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts", "experiment")
    artifact_path = store.path("indexes/stuffing/index-current.json")
    artifact = IndexArtifact(
        id="index-current",
        system_id="stuffing",
        corpus_id="corpus",
        artifact_path=str(artifact_path),
    )
    store.write_json("indexes/stuffing/index-current.json", artifact)

    assert _load_index_artifacts_if_present(store) == [artifact]


def test_interrupted_resume_preserves_not_yet_rebuilt_artifacts(tmp_path) -> None:
    config = _foundry_export_smoke_config(tmp_path).model_copy(
        update={"systems": ["stuffing", "map_reduce"]}
    )
    _prepare(config)
    asyncio.run(_index(config))

    store = ArtifactStore(config.storage.artifacts_dir, config.experiment.id)
    artifacts = _load_index_artifacts_if_present(store)
    assert {artifact.system_id for artifact in artifacts} == {"stuffing", "map_reduce"}
    for artifact in artifacts:
        stale = artifact.model_copy(
            update={
                "build_metadata": {
                    **artifact.build_metadata,
                    "index_config_signature": "stale-index-config",
                }
            }
        )
        store.write_json(Path(stale.artifact_path).relative_to(store.experiment_dir), stale)

    resume = config.model_copy(update={"run_control": RunControlConfig(resume=True)})
    budget = BudgetLedger(RunControlConfig(max_model_calls=1))
    with pytest.raises(BudgetExceeded, match="Budget exceeded after index stuffing"):
        asyncio.run(_index(resume, budget=budget))

    recovered = _load_index_artifacts_if_present(store)
    assert {artifact.system_id for artifact in recovered} == {"stuffing", "map_reduce"}


def test_query_records_are_reusable_only_for_matching_index_signature(tmp_path) -> None:
    artifact_path = tmp_path / "index.json"
    map_path = tmp_path / "map.json"
    artifact_path.write_text("{}", encoding="utf-8")
    map_path.write_text("{}", encoding="utf-8")
    artifact = _index_artifact(
        artifact_path=artifact_path,
        map_path=map_path,
        prompt_safety_policy=PROMPT_SAFETY_POLICY_VERSION,
        source_reference_normalization_policy=(MAP_SOURCE_REFERENCE_NORMALIZATION_POLICY_VERSION),
    )
    record = RagRunRecord(
        run_id="run",
        experiment_id="exp",
        system_id="stuffing",
        corpus_id="corpus",
        item_id="item",
        selected_document_ids=[],
        retrieved_items=[],
        answer="answer",
        citations=[],
        status="succeeded",
        index_artifact_id=artifact.id,
        index_artifact_signature=index_artifact_signature(artifact),
        query_policy_version=QUERY_RUN_POLICY_VERSION,
        query_config_signature="query-v1",
    )

    assert (
        _is_reusable_query_record(
            record,
            artifact,
            expected_query_config_signature="query-v1",
        )
        is True
    )
    assert (
        _is_reusable_query_record(
            record.model_copy(update={"query_policy_version": "old-query-policy"}),
            artifact,
            expected_query_config_signature="query-v1",
        )
        is False
    )
    assert (
        _is_reusable_query_record(
            record.model_copy(update={"index_artifact_signature": "old-signature"}),
            artifact,
            expected_query_config_signature="query-v1",
        )
        is False
    )
    assert (
        _is_reusable_query_record(
            record,
            artifact,
            expected_query_config_signature="query-v2",
        )
        is False
    )


def test_exhausted_query_budget_writes_skipped_records(tmp_path) -> None:
    config = _foundry_export_smoke_config(tmp_path)
    budget_config = config.model_copy(update={"run_control": RunControlConfig(max_model_calls=0)})

    _prepare(config)
    asyncio.run(_index(config))
    asyncio.run(_query(budget_config))

    experiment_dir = tmp_path / "artifacts" / "foundry-eval-export-smoke"
    records = _run_records(experiment_dir / "runs/stuffing.jsonl")
    query_usage = [
        UsageEvent.model_validate_json(line)
        for line in (experiment_dir / "costs/query-usage.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert len(records) == 2
    assert {record.status for record in records} == {"skipped"}
    assert all("Budget exhausted before query stuffing" in str(record.error) for record in records)
    assert query_usage == []


def test_index_budget_persists_completed_artifact_before_raising(tmp_path) -> None:
    config = _foundry_export_smoke_config(tmp_path).model_copy(
        update={"run_control": RunControlConfig(max_model_calls=1)}
    )

    _prepare(config)
    with pytest.raises(BudgetExceeded, match="Budget exceeded after index stuffing"):
        asyncio.run(_index(config))

    experiment_dir = tmp_path / "artifacts" / "foundry-eval-export-smoke"
    assert _jsonl_count(experiment_dir / "indexes/index_artifacts.jsonl") == 1
    assert _jsonl_count(experiment_dir / "costs/index-usage.jsonl") == 2


def test_usage_merge_keeps_distinct_retry_attempts_without_duplicating_writes() -> None:
    first = UsageEvent(
        experiment_id="experiment",
        run_id="run",
        system_id="stuffing",
        corpus_id="corpus",
        stage="index_system",
        kind="system",
        input_tokens=100,
        timestamp="2026-09-15T08:00:00+00:00",
    )
    retry = first.model_copy(
        update={
            "input_tokens": 120,
            "timestamp": "2026-09-15T09:00:00+00:00",
        }
    )

    merged = _merge_usage_events([first], [retry])
    assert len(merged) == 2
    assert sum(event.input_tokens for event in merged) == 220
    assert _merge_usage_events(merged, [retry]) == merged


def _foundry_export_smoke_config(tmp_path: Path):
    config = load_experiment_config(
        Path("configs/experiments/foundry-eval-export-smoke.yaml"),
        project_root=Path.cwd(),
    )
    return config.model_copy(
        update={
            "storage": config.storage.model_copy(update={"artifacts_dir": tmp_path / "artifacts"})
        }
    )


def _index_artifact(
    *,
    artifact_path: Path,
    map_path: Path,
    prompt_safety_policy: str,
    source_reference_normalization_policy: str,
) -> IndexArtifact:
    return IndexArtifact(
        id="index-current",
        system_id="stuffing",
        corpus_id="corpus",
        artifact_path=str(artifact_path),
        document_map_ids=["map-current"],
        build_metadata={
            "document_map_paths": {"map-current": str(map_path)},
            "prompt_safety_policy": prompt_safety_policy,
            "source_reference_normalization_policy": source_reference_normalization_policy,
        },
    )


def _document_map() -> DocumentMap:
    return DocumentMap(
        document_id="document",
        overview="Overview",
        entries=[
            MapEntry(
                id="entry",
                kind="fact",
                label="Label",
                summary="Summary",
                source_references=[
                    SourceReference(document_id="document", segment_ids=["segment"])
                ],
                attributes={"zeta": 1, "alpha": 2},
            )
        ],
        facets={"zeta": 1, "alpha": 2},
        construction_method="stuffing",
    )


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def _run_records(path: Path) -> list[RagRunRecord]:
    return [
        RagRunRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
