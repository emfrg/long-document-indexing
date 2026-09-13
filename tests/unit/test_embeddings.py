from __future__ import annotations

from types import SimpleNamespace

import pytest

from long_document_indexing.models.embeddings import OpenAICompatibleEmbeddingClient


async def test_openai_compatible_embedding_client_preserves_input_order() -> None:
    embeddings_api = _FakeEmbeddingsApi()
    client = OpenAICompatibleEmbeddingClient(
        base_url="https://example.openai.azure.com/openai/v1/",
        deployment="text-embedding-3-small",
        api_key="test-key",
        dimensions=2,
        client=SimpleNamespace(embeddings=embeddings_api),
    )

    response = await client.embed(["first", "second"])

    assert client.base_url == "https://example.openai.azure.com/openai/v1/"
    assert response.embeddings == [[1.0, 0.0], [0.0, 1.0]]
    assert response.usage.input_tokens == 4
    assert response.usage.model_calls == 1
    assert embeddings_api.kwargs == {
        "model": "text-embedding-3-small",
        "input": ["first", "second"],
        "encoding_format": "float",
        "dimensions": 2,
    }


async def test_openai_compatible_embedding_client_rejects_missing_vectors() -> None:
    client = OpenAICompatibleEmbeddingClient(
        base_url="https://example.openai.azure.com/openai/v1/",
        deployment="text-embedding-3-small",
        api_key="test-key",
        client=SimpleNamespace(
            embeddings=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(data=[], usage=None)
            )
        ),
    )

    with pytest.raises(RuntimeError, match="count mismatch"):
        await client.embed(["missing"])


def test_openai_compatible_embedding_client_normalizes_endpoint_route() -> None:
    client = OpenAICompatibleEmbeddingClient(
        base_url="https://example.openai.azure.com/openai/v1/responses",
        deployment="text-embedding-3-small",
        api_key="test-key",
    )

    assert client.base_url == "https://example.openai.azure.com/openai/v1/"


class _FakeEmbeddingsApi:
    def __init__(self) -> None:
        self.kwargs = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            model="text-embedding-3-small",
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ],
            usage=SimpleNamespace(prompt_tokens=4, total_tokens=4),
        )
