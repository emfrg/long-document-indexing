from __future__ import annotations

import os

import pytest

from long_document_indexing.storage.artifacts import ArtifactStore


def test_atomic_write_preserves_previous_file_when_replace_fails(tmp_path, monkeypatch) -> None:
    store = ArtifactStore(tmp_path / "artifacts", "experiment")
    path = store.write_json("state.json", {"version": 1})
    original = path.read_text(encoding="utf-8")

    def fail_replace(source, destination) -> None:
        del source, destination
        raise OSError("simulated interruption")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated interruption"):
        store.write_json("state.json", {"version": 2})

    assert path.read_text(encoding="utf-8") == original
    assert list(path.parent.glob(".state.json.*.tmp")) == []


def test_jsonl_write_replaces_complete_file(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts", "experiment")

    path = store.write_jsonl("records.jsonl", [{"id": 1}])
    store.write_jsonl("records.jsonl", [{"id": 2}, {"id": 3}])

    assert store.read_jsonl("records.jsonl") == [{"id": 2}, {"id": 3}]
    assert path.read_text(encoding="utf-8").endswith("\n")
