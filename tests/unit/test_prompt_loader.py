from __future__ import annotations

import pytest

from long_document_indexing.prompts import PromptLoader


def test_prompt_loader_loads_markdown_by_stem(tmp_path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "answer.md").write_text("Answer from evidence.", encoding="utf-8")

    loader = PromptLoader(prompt_dir)

    assert loader.load("answer") == "Answer from evidence."


def test_prompt_loader_rejects_path_escape(tmp_path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()

    loader = PromptLoader(prompt_dir)

    with pytest.raises(ValueError):
        loader.resolve("..", "outside")
