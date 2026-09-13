from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from long_document_indexing.domain.corpus import Corpus
from long_document_indexing.domain.runs import RetrievedItem, UsageRecord
from long_document_indexing.models.base import EmbeddingClient

DENSE_INDEX_VERSION = "dense-vector/v1"


class DenseVectorBackend:
    """Persistent dense-vector retrieval over canonical source segments."""

    def __init__(
        self,
        artifact_dir: Path,
        embedding_client: EmbeddingClient,
        *,
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError("embedding batch size must be positive")
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_client = embedding_client
        self.batch_size = batch_size
        self._indexes: dict[str, dict[str, Any]] = {}
        self._pending_usage = UsageRecord()

    async def index(self, corpus: Corpus) -> str:
        self._pending_usage = UsageRecord()
        index_id = _index_id(corpus, self.embedding_client.model_id)
        path = self._path(index_id)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            _validate_index(payload, expected_model_id=self.embedding_client.model_id)
            self._indexes[index_id] = payload
            return index_id

        records = [
            {
                "document_id": document.id,
                "segment_id": segment.id,
                "text": segment.text,
                "metadata": segment.metadata,
            }
            for document in corpus.documents
            for segment in sorted(document.segments, key=lambda item: item.order)
        ]
        embeddings: list[list[float]] = []
        usage_records: list[UsageRecord] = []
        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            response = await self.embedding_client.embed([str(record["text"]) for record in batch])
            embeddings.extend(response.embeddings)
            usage_records.append(response.usage)

        if len(embeddings) != len(records):
            raise RuntimeError(
                f"dense index embedding count mismatch: {len(records)} records, "
                f"{len(embeddings)} embeddings"
            )
        for record, embedding in zip(records, embeddings, strict=True):
            record["embedding"] = embedding
            record["norm"] = _norm(embedding)

        payload = {
            "id": index_id,
            "version": DENSE_INDEX_VERSION,
            "corpus_id": corpus.id,
            "embedding_model_id": self.embedding_client.model_id,
            "records": records,
        }
        path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
        self._indexes[index_id] = payload
        self._pending_usage = _combine_usage(usage_records)
        return index_id

    async def search(
        self,
        index_id: str,
        query: str,
        *,
        document_ids: list[str] | None,
        top_k: int,
    ) -> list[RetrievedItem]:
        self._pending_usage = UsageRecord()
        if top_k < 1:
            return []

        index = self._load(index_id)
        response = await self.embedding_client.embed([query])
        self._pending_usage = response.usage
        if len(response.embeddings) != 1:
            raise RuntimeError("query embedding response must contain exactly one vector")
        query_embedding = response.embeddings[0]
        query_norm = _norm(query_embedding)
        allowed = set(document_ids) if document_ids is not None else None

        scored = []
        for record in index["records"]:
            if allowed is not None and record["document_id"] not in allowed:
                continue
            embedding = [float(value) for value in record["embedding"]]
            score = _cosine(query_embedding, query_norm, embedding, float(record["norm"]))
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
                document_id=str(record["document_id"]),
                segment_id=str(record["segment_id"]),
                text=str(record["text"]),
                score=score,
                rank=rank,
                retrieval_stage="dense_vector",
                metadata=dict(record.get("metadata", {})),
            )
            for rank, (score, record) in enumerate(scored[:top_k], start=1)
        ]

    def artifact_path(self, index_id: str) -> Path | None:
        path = self._path(index_id)
        return path if path.exists() else None

    def consume_usage(self) -> UsageRecord:
        usage = self._pending_usage
        self._pending_usage = UsageRecord()
        return usage

    def _path(self, index_id: str) -> Path:
        return self.artifact_dir / f"{index_id}.json"

    def _load(self, index_id: str) -> dict[str, Any]:
        if index_id not in self._indexes:
            path = self._path(index_id)
            if not path.exists():
                raise FileNotFoundError(f"dense retrieval index not found: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            _validate_index(payload, expected_model_id=self.embedding_client.model_id)
            self._indexes[index_id] = payload
        return self._indexes[index_id]


def _index_id(corpus: Corpus, embedding_model_id: str) -> str:
    payload = {
        "version": DENSE_INDEX_VERSION,
        "embedding_model_id": embedding_model_id,
        "corpus": corpus.model_dump(mode="json"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"dense-vector-{digest[:16]}"


def _validate_index(payload: dict[str, Any], *, expected_model_id: str) -> None:
    if payload.get("version") != DENSE_INDEX_VERSION:
        raise ValueError("dense retrieval index has an unsupported version")
    if payload.get("embedding_model_id") != expected_model_id:
        raise ValueError("dense retrieval index embedding model does not match configuration")


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine(
    left: list[float],
    left_norm: float,
    right: list[float],
    right_norm: float,
) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"embedding dimension mismatch: query has {len(left)}, index has {len(right)}"
        )
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _combine_usage(records: list[UsageRecord]) -> UsageRecord:
    estimated_costs = [
        record.estimated_cost for record in records if record.estimated_cost is not None
    ]
    return UsageRecord(
        input_tokens=sum(record.input_tokens for record in records),
        output_tokens=sum(record.output_tokens for record in records),
        model_calls=sum(record.model_calls for record in records),
        tool_calls=sum(record.tool_calls for record in records),
        duration_ms=sum(record.duration_ms for record in records),
        estimated_cost=sum(estimated_costs) if estimated_costs else None,
    )
