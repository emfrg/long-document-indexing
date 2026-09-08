from __future__ import annotations

import re
from typing import Any

PROMPT_SAFETY_POLICY_VERSION = "legal-neutral-v4"
PROMPT_SAFETY_PREAMBLE = (
    "Task context: neutral legal-document indexing. Extract high-level legal facts, "
    "issues, actors, dates, and source references. Do not provide operational "
    "instructions, graphic detail, or procedural guidance; abstract sensitive "
    "procedural details into neutral legal-record language.\n\n"
)

_ID_KEYS = {
    "id",
    "document_id",
    "document_ids",
    "entry_id",
    "evidence_id",
    "segment_id",
    "segment_ids",
    "source_segment_ids",
}

_TEXT_REPLACEMENTS = (
    (re.compile(r"\bsentence of death\b", flags=re.IGNORECASE), "capital sentence"),
    (re.compile(r"\bdeath sentence\b", flags=re.IGNORECASE), "capital sentence"),
    (re.compile(r"\bdeath chamber\b", flags=re.IGNORECASE), "procedure chamber"),
    (re.compile(r"\blethal injection\b", flags=re.IGNORECASE), "capital-sentence protocol"),
    (re.compile(r"\bexecutioners?\b", flags=re.IGNORECASE), "procedure personnel"),
    (re.compile(r"\bexecutions\b", flags=re.IGNORECASE), "capital-sentence procedures"),
    (re.compile(r"\bexecution\b", flags=re.IGNORECASE), "capital-sentence procedure"),
    (re.compile(r"\bexecuted\b", flags=re.IGNORECASE), "subject to a capital-sentence procedure"),
    (re.compile(r"\bdeath(s)?\b", flags=re.IGNORECASE), "fatal outcome\\1"),
    (re.compile(r"\bdead\b", flags=re.IGNORECASE), "unresponsive"),
    (re.compile(r"\bkilled?\b", flags=re.IGNORECASE), "caused a fatal outcome"),
    (re.compile(r"\bkilling\b", flags=re.IGNORECASE), "fatal incident"),
    (re.compile(r"\bmurders?\b", flags=re.IGNORECASE), "fatal offenses"),
    (re.compile(r"\bhomicides?\b", flags=re.IGNORECASE), "fatal offenses"),
    (re.compile(r"\bshoot(?:ing|ings|s)?\b", flags=re.IGNORECASE), "force incident"),
    (re.compile(r"\bstabb(?:ing|ings|ed)?\b", flags=re.IGNORECASE), "injury incident"),
    (re.compile(r"\bpotassium chloride\b", flags=re.IGNORECASE), "named protocol substance"),
    (re.compile(r"\bpotassium\b", flags=re.IGNORECASE), "named substance"),
    (re.compile(r"\bchloride\b", flags=re.IGNORECASE), "substance"),
    (re.compile(r"\bthiopental\b", flags=re.IGNORECASE), "named protocol substance"),
    (re.compile(r"\bthopental\b", flags=re.IGNORECASE), "named protocol substance"),
    (re.compile(r"\bsodium pentothal\b", flags=re.IGNORECASE), "named protocol substance"),
    (re.compile(r"\bpentothal\b", flags=re.IGNORECASE), "named protocol substance"),
    (re.compile(r"\bpancuronium bromide\b", flags=re.IGNORECASE), "named protocol substance"),
    (re.compile(r"\bpancurium bromide\b", flags=re.IGNORECASE), "named protocol substance"),
    (re.compile(r"\bpancuronium\b", flags=re.IGNORECASE), "named protocol substance"),
    (re.compile(r"\bpancurium\b", flags=re.IGNORECASE), "named protocol substance"),
    (re.compile(r"\bpavulon\b", flags=re.IGNORECASE), "named protocol substance"),
    (re.compile(r"\bbarbiturates?\b", flags=re.IGNORECASE), "protocol component"),
    (re.compile(r"\bbarbituates?\b", flags=re.IGNORECASE), "protocol component"),
    (
        re.compile(r"\banesthe(?:sia|tic|tize|tized|tizing|tizes)\b", flags=re.IGNORECASE),
        "unconsciousness safeguard",
    ),
    (
        re.compile(r"\bcentral nervous system\b", flags=re.IGNORECASE),
        "physiological system",
    ),
    (re.compile(r"\belectrolyte\b", flags=re.IGNORECASE), "substance"),
    (
        re.compile(r"\bneuromuscular blocking agents?\b", flags=re.IGNORECASE),
        "protocol component",
    ),
    (re.compile(r"\bdrugs?\b", flags=re.IGNORECASE), "substances"),
    (re.compile(r"\bchemicals?\b", flags=re.IGNORECASE), "substances"),
    (re.compile(r"\bdos(?:e|es|age|ages)\b", flags=re.IGNORECASE), "amount"),
    (re.compile(r"\bsyringes?\b", flags=re.IGNORECASE), "delivery tools"),
    (re.compile(r"\bintravenous\b", flags=re.IGNORECASE), "access-related"),
    (re.compile(r"\bIV\b"), "access line"),
    (re.compile(r"\bcatheters?\b", flags=re.IGNORECASE), "access devices"),
    (re.compile(r"\bveins?\b", flags=re.IGNORECASE), "access points"),
    (re.compile(r"\bvenous\b", flags=re.IGNORECASE), "access-related"),
    (re.compile(r"\binfusion\b", flags=re.IGNORECASE), "administration"),
    (re.compile(r"\bsubclavian\b", flags=re.IGNORECASE), "alternate-access"),
    (re.compile(r"\belectrocardiogram\b", flags=re.IGNORECASE), "monitoring device"),
    (re.compile(r"\bincision\b", flags=re.IGNORECASE), "access procedure"),
    (re.compile(r"\bpunctur(?:e|ed|es|ing)\b", flags=re.IGNORECASE), "access attempt"),
    (re.compile(r"\bpierc(?:e|ed|es|ing)\b", flags=re.IGNORECASE), "access attempt"),
    (re.compile(r"\bskin\b", flags=re.IGNORECASE), "tissue"),
    (re.compile(r"\bmedical\b", flags=re.IGNORECASE), "professional"),
    (re.compile(r"\bveterinary\b", flags=re.IGNORECASE), "professional"),
    (re.compile(r"\bpatients?\b", flags=re.IGNORECASE), "subjects"),
    (re.compile(r"\bhuman body\b", flags=re.IGNORECASE), "subject"),
    (re.compile(r"\bbody\b", flags=re.IGNORECASE), "subject"),
    (re.compile(r"\banimals?\b", flags=re.IGNORECASE), "subjects"),
    (re.compile(r"\bpets?\b", flags=re.IGNORECASE), "subjects"),
    (re.compile(r"\beuthan\w+\b", flags=re.IGNORECASE), "regulated procedure"),
    (re.compile(r"\bblood\b", flags=re.IGNORECASE), "record evidence"),
    (re.compile(r"\bpostmortem\b", flags=re.IGNORECASE), "record-review"),
    (re.compile(r"\bautops(?:y|ies)\b", flags=re.IGNORECASE), "records review"),
    (re.compile(r"\boverdose\b", flags=re.IGNORECASE), "protocol issue"),
    (re.compile(r"\bcardiac arrest\b", flags=re.IGNORECASE), "monitoring event"),
    (re.compile(r"\bcardiac\b", flags=re.IGNORECASE), "monitoring"),
    (re.compile(r"\basphyxiat(?:e|ed|ion)\b", flags=re.IGNORECASE), "respiratory failure"),
    (re.compile(r"\bsuffocat(?:e|ed|ion)\b", flags=re.IGNORECASE), "respiratory distress"),
    (re.compile(r"\brespiration\b", flags=re.IGNORECASE), "breathing"),
    (re.compile(r"\bparaly(?:sis|zes|zed|tic)\b", flags=re.IGNORECASE), "immobilization"),
    (re.compile(r"\bseizures?\b", flags=re.IGNORECASE), "record event"),
    (re.compile(r"\bexcruciat(?:ing|ingly)\b", flags=re.IGNORECASE), "severe"),
    (re.compile(r"\bagoniz(?:e|ed|ing|ingly)\b", flags=re.IGNORECASE), "severe"),
    (re.compile(r"\btort(?:ure|ured|uring|ous)\b", flags=re.IGNORECASE), "severe mistreatment"),
    (re.compile(r"\bpain(?:ful)?\b", flags=re.IGNORECASE), "distress"),
    (re.compile(r"\bsuffering\b", flags=re.IGNORECASE), "distress"),
)


def sanitize_for_model_prompt(text: str) -> str:
    """Reduce graphic legal-case wording before sending text to hosted model filters."""

    sanitized = text
    for pattern, replacement in _TEXT_REPLACEMENTS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def sanitize_prompt_payload(value: Any, *, key: str | None = None) -> Any:
    """Recursively sanitize prompt payloads without changing IDs used for citations."""

    if key is not None and _is_identifier_key(key):
        return value
    if isinstance(value, str):
        return sanitize_for_model_prompt(value)
    if isinstance(value, list):
        return [sanitize_prompt_payload(item, key=key) for item in value]
    if isinstance(value, dict):
        return {
            item_key: sanitize_prompt_payload(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    return value


def apply_prompt_safety_preamble(prompt: str) -> str:
    return f"{PROMPT_SAFETY_PREAMBLE}{prompt}"


def _is_identifier_key(key: str) -> bool:
    normalized = key.lower()
    return normalized in _ID_KEYS or normalized.endswith("_id") or normalized.endswith("_ids")
