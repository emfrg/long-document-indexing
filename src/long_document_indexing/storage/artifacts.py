from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class ArtifactStore:
    """Filesystem-backed storage for one experiment's inspectable artifacts."""

    def __init__(self, root: Path, experiment_id: str) -> None:
        self.root = root
        self.experiment_id = experiment_id
        self.experiment_dir = self.root / experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str | Path) -> Path:
        path = self.experiment_dir.joinpath(*map(Path, parts))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative_path: str | Path, payload: BaseModel | dict[str, Any]) -> Path:
        path = self.path(relative_path)
        data = _to_jsonable(payload)
        atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")
        return path

    def read_json(self, relative_path: str | Path) -> dict[str, Any]:
        path = self.path(relative_path)
        return json.loads(path.read_text(encoding="utf-8"))

    def write_jsonl(
        self,
        relative_path: str | Path,
        records: Iterable[BaseModel | dict[str, Any]],
    ) -> Path:
        path = self.path(relative_path)
        lines = [json.dumps(_to_jsonable(record), sort_keys=True) for record in records]
        atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
        return path

    def read_jsonl(self, relative_path: str | Path) -> list[dict[str, Any]]:
        path = self.path(relative_path)
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def _to_jsonable(payload: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    return payload


def atomic_write_text(path: Path, content: str) -> None:
    """Replace a text file only after its complete content reaches disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
