from __future__ import annotations

import json
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
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
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
