from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_LEXICAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "why",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def approximate_token_count(text: str) -> int:
    return len(tokenize(text))


def summarize_text(text: str, *, max_chars: int = 260) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 3]}..."


def lexical_similarity(left: str, right: str) -> float:
    left_counts = Counter(_content_tokens(left))
    right_counts = Counter(_content_tokens(right))
    if not left_counts or not right_counts:
        return 0.0

    shared = left_counts.keys() & right_counts.keys()
    dot = sum(left_counts[token] * right_counts[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _content_tokens(text: str) -> list[str]:
    tokens = tokenize(text)
    filtered = [token for token in tokens if token not in _LEXICAL_STOPWORDS]
    return filtered or tokens
