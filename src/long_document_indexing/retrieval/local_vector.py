from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from long_document_indexing.domain.corpus import Corpus
from long_document_indexing.domain.runs import RetrievedItem
from long_document_indexing.text import tokenize


class LocalVectorBackend:
    """Small lexical vector backend for local deterministic benchmarks."""

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._indexes: dict[str, dict[str, Any]] = {}

    async def index(self, corpus: Corpus) -> str:
        index_id = _index_id(corpus)
        path = self._path(index_id)
        if path.exists():
            self._indexes[index_id] = json.loads(path.read_text(encoding="utf-8"))
            return index_id

        records = []
        document_frequencies: Counter[str] = Counter()
        term_counts_by_record: list[Counter[str]] = []

        for document in corpus.documents:
            for segment in sorted(document.segments, key=lambda item: item.order):
                term_counts = Counter(tokenize(segment.text))
                term_counts_by_record.append(term_counts)
                document_frequencies.update(term_counts.keys())
                records.append(
                    {
                        "document_id": document.id,
                        "segment_id": segment.id,
                        "text": segment.text,
                        "metadata": segment.metadata,
                    }
                )

        idf = {
            term: math.log((len(records) + 1) / (frequency + 1)) + 1.0
            for term, frequency in document_frequencies.items()
        }

        for record, term_counts in zip(records, term_counts_by_record, strict=True):
            weights = _weights(term_counts, idf)
            record["weights"] = weights
            record["norm"] = _norm(weights)

        payload = {"id": index_id, "corpus_id": corpus.id, "idf": idf, "records": records}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._indexes[index_id] = payload
        return index_id

    async def search(
        self,
        index_id: str,
        query: str,
        *,
        document_ids: list[str] | None,
        top_k: int,
    ) -> list[RetrievedItem]:
        if top_k < 1:
            return []

        index = self._load(index_id)
        allowed = set(document_ids) if document_ids is not None else None
        query_weights = _weights(Counter(tokenize(query)), index["idf"])
        query_norm = _norm(query_weights)

        scored = []
        for record in index["records"]:
            if allowed is not None and record["document_id"] not in allowed:
                continue
            score = _cosine(query_weights, query_norm, record["weights"], record["norm"])
            scored.append((score, record))

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1]["document_id"],
                item[1]["segment_id"],
            )
        )

        return [
            RetrievedItem(
                document_id=record["document_id"],
                segment_id=record["segment_id"],
                text=record["text"],
                score=score,
                rank=rank,
                retrieval_stage="local_vector",
                metadata=record.get("metadata", {}),
            )
            for rank, (score, record) in enumerate(scored[:top_k], start=1)
        ]

    def artifact_path(self, index_id: str) -> Path | None:
        path = self._path(index_id)
        return path if path.exists() else None

    def _path(self, index_id: str) -> Path:
        return self.artifact_dir / f"{index_id}.json"

    def _load(self, index_id: str) -> dict[str, Any]:
        if index_id not in self._indexes:
            path = self._path(index_id)
            if not path.exists():
                raise FileNotFoundError(f"retrieval index not found: {path}")
            self._indexes[index_id] = json.loads(path.read_text(encoding="utf-8"))
        return self._indexes[index_id]


def _weights(term_counts: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    return {term: count * idf.get(term, 0.0) for term, count in term_counts.items()}


def _norm(weights: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in weights.values()))


def _cosine(
    left: dict[str, float],
    left_norm: float,
    right: dict[str, float],
    right_norm: float,
) -> float:
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    shared_terms = left.keys() & right.keys()
    dot = sum(left[term] * right[term] for term in shared_terms)
    return dot / (left_norm * right_norm)


def _index_id(corpus: Corpus) -> str:
    payload = corpus.model_dump(mode="json")
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"local-vector-{digest[:16]}"
