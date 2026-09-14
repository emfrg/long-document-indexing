from __future__ import annotations

from long_document_indexing.domain.corpus import Corpus, Document, Segment
from long_document_indexing.domain.runs import UsageRecord
from long_document_indexing.models.base import EmbeddingResponse
from long_document_indexing.retrieval.dense_vector import DenseVectorBackend


async def test_dense_vector_uses_embeddings_and_honors_document_filter(tmp_path) -> None:
    embedding_client = _SemanticEmbeddingClient()
    progress: list[str] = []
    backend = DenseVectorBackend(
        tmp_path,
        embedding_client,
        batch_size=8,
        progress=progress.append,
    )
    index_id = await backend.index(_corpus())

    index_usage = backend.consume_usage()
    assert index_usage.model_calls == 1
    assert backend.artifact_path(index_id) is not None
    assert progress == ["dense_vector embedding corpus: batch 1/1 (2 segment(s))"]

    results = await backend.search(
        index_id,
        "ability to pay",
        document_ids=None,
        top_k=2,
    )

    assert [item.segment_id for item in results] == ["legal_s1", "business_s1"]
    assert all(item.retrieval_stage == "dense_vector" for item in results)
    assert backend.consume_usage().model_calls == 1
    assert progress[-1] == "dense_vector embedding query"

    filtered = await backend.search(
        index_id,
        "ability to pay",
        document_ids=["doc_business"],
        top_k=2,
    )

    assert [item.document_id for item in filtered] == ["doc_business"]


async def test_dense_vector_reuses_persisted_embeddings(tmp_path) -> None:
    first_client = _SemanticEmbeddingClient()
    first_backend = DenseVectorBackend(tmp_path, first_client)
    first_id = await first_backend.index(_corpus())
    assert first_client.calls == 1

    second_client = _SemanticEmbeddingClient()
    progress: list[str] = []
    second_backend = DenseVectorBackend(tmp_path, second_client, progress=progress.append)
    second_id = await second_backend.index(_corpus())

    assert second_id == first_id
    assert second_client.calls == 0
    assert second_backend.consume_usage().model_calls == 0
    assert progress == ["dense_vector reused embeddings corpus (2 segment(s))"]


class _SemanticEmbeddingClient:
    model_id = "semantic-test-v1"

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        self.calls += 1
        embeddings = [
            [1.0, 0.0] if "indigent" in text.lower() or "ability to pay" in text.lower()
            else [0.0, 1.0]
            for text in texts
        ]
        return EmbeddingResponse(
            embeddings=embeddings,
            usage=UsageRecord(input_tokens=len(texts), model_calls=1),
        )


def _corpus() -> Corpus:
    return Corpus(
        id="corpus",
        documents=[
            Document(
                id="doc_legal",
                corpus_id="corpus",
                segments=[
                    Segment(
                        id="legal_s1",
                        document_id="doc_legal",
                        order=1,
                        text="The court considered whether the indigent debtor could afford it.",
                    )
                ],
            ),
            Document(
                id="doc_business",
                corpus_id="corpus",
                segments=[
                    Segment(
                        id="business_s1",
                        document_id="doc_business",
                        order=1,
                        text="The companies completed a corporate merger.",
                    )
                ],
            ),
        ],
    )
