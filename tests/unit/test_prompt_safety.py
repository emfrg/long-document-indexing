from __future__ import annotations

from long_document_indexing.prompt_safety import (
    apply_prompt_safety_preamble,
    sanitize_for_model_prompt,
    sanitize_for_model_recovery_prompt,
    sanitize_prompt_payload,
)


def test_sanitize_for_model_prompt_uses_neutral_legal_terms() -> None:
    text = "The lethal injection execution caused excruciating pain, torture, and death."

    sanitized = sanitize_for_model_prompt(text)

    assert "lethal injection" not in sanitized.lower()
    assert "execution" not in sanitized.lower()
    assert "excruciating" not in sanitized.lower()
    assert "torture" not in sanitized.lower()
    assert sanitized == (
        "The capital-sentence protocol capital-sentence procedure caused "
        "severe distress, severe mistreatment, and fatal outcome."
    )


def test_sanitize_prompt_payload_preserves_citation_identifiers() -> None:
    payload = {
        "document_id": "case:execution:doc_0001",
        "segment_ids": ["case:execution:seg_0001"],
        "summary": "Autopsy reports discuss botched executions and postmortem blood evidence.",
        "children": [
            {
                "id": "execution_entry",
                "label": "Lethal injection execution evidence",
            }
        ],
    }

    sanitized = sanitize_prompt_payload(payload)

    assert sanitized["document_id"] == "case:execution:doc_0001"
    assert sanitized["segment_ids"] == ["case:execution:seg_0001"]
    assert "execution" not in sanitized["summary"].lower()
    assert "autopsy" not in sanitized["summary"].lower()
    assert sanitized["children"][0]["id"] == "execution_entry"
    assert "lethal injection" not in sanitized["children"][0]["label"].lower()


def test_apply_prompt_safety_preamble_marks_neutral_legal_indexing_context() -> None:
    prompt = apply_prompt_safety_preamble("Build a document map.")

    assert prompt.startswith("Task context: neutral legal-document indexing.")
    assert prompt.endswith("Build a document map.")


def test_recovery_prompt_abstracts_sensitive_legal_allegations() -> None:
    text = (
        "The complaint alleges sexual harassment, sexual propositions, offensive touching "
        "involving her breasts and backside, and the remark \"I'm horny.\""
    )

    sanitized = sanitize_for_model_recovery_prompt(text)

    assert sanitized.startswith("Recovery instruction:")
    assert "sexual" not in sanitized.lower()
    assert "breast" not in sanitized.lower()
    assert "backside" not in sanitized.lower()
    assert "horny" not in sanitized.lower()
    assert "workplace harassment based on sex" in sanitized
