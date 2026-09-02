from __future__ import annotations

from pathlib import Path


class PromptLoader:
    """Loads versioned prompt files from the repository prompt directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def load(self, *parts: str) -> str:
        path = self.resolve(*parts)
        return path.read_text(encoding="utf-8")

    def resolve(self, *parts: str) -> Path:
        if not parts:
            raise ValueError("at least one prompt path part is required")

        path = self.root.joinpath(*parts)
        if path.suffix == "":
            path = path.with_suffix(".md")
        path = path.resolve()

        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"prompt path escapes prompt root: {path}") from exc

        if not path.exists():
            raise FileNotFoundError(f"prompt file not found: {path}")
        return path
