from __future__ import annotations

from types import SimpleNamespace

import pytest

from long_document_indexing.domain.maps import DocumentMap
from long_document_indexing.models.base import GenerationRequest
from long_document_indexing.models.openai_compatible import OpenAICompatibleTextGenerationClient
from long_document_indexing.models.structured_outputs import (
    StructuredDocumentMap,
    StructuredMapAttribute,
    StructuredMapEntry,
    StructuredSourceReference,
)


async def test_openai_compatible_client_normalizes_chat_completion_response() -> None:
    completions = _FakeCompletions()
    client = OpenAICompatibleTextGenerationClient(
        base_url="https://example.openai.azure.com/openai/v1/",
        deployment="doc-map",
        api_key="test-key",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    response = await client.generate(
        GenerationRequest(
            prompt="Return JSON.",
            prompt_name="stuffing/map",
            metadata={"task": "document_map"},
        )
    )

    assert response.content == '{"ok": true}'
    assert response.usage.input_tokens == 7
    assert response.usage.output_tokens == 3
    assert response.usage.model_calls == 1
    assert response.metadata["deployment"] == "doc-map"
    assert "api_key" not in response.metadata
    assert completions.kwargs["model"] == "doc-map"
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert completions.kwargs["temperature"] == 0.0
    assert response.metadata["api"] == "chat_completions"


async def test_openai_compatible_client_normalizes_responses_api_response() -> None:
    responses = _FakeResponses()
    client = OpenAICompatibleTextGenerationClient(
        base_url="https://example.openai.azure.com/openai/v1/responses",
        deployment="gpt-5-mini-doc-map-generator",
        api="responses",
        api_key="test-key",
        max_output_tokens=256,
        client=SimpleNamespace(responses=responses),
    )

    response = await client.generate(
        GenerationRequest(
            prompt="Return JSON.",
            prompt_name="stuffing/map",
            metadata={"task": "document_map"},
        )
    )

    assert client.base_url == "https://example.openai.azure.com/openai/v1/"
    assert response.content == '{"ok": true}'
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 5
    assert response.usage.model_calls == 1
    assert response.metadata["api"] == "responses"
    assert response.metadata["deployment"] == "gpt-5-mini-doc-map-generator"
    assert responses.kwargs["model"] == "gpt-5-mini-doc-map-generator"
    assert responses.kwargs["input"] == [{"role": "user", "content": "Return JSON."}]
    assert responses.kwargs["max_output_tokens"] == 256
    assert responses.kwargs["text"] == {"format": {"type": "json_object"}}
    assert "temperature" not in responses.kwargs


async def test_openai_compatible_client_parses_structured_responses_api_response() -> None:
    responses = _FakeResponses()
    client = OpenAICompatibleTextGenerationClient(
        base_url="https://example.openai.azure.com/openai/v1/responses",
        deployment="gpt-5-mini-doc-map-generator",
        api="responses",
        api_key="test-key",
        max_output_tokens=256,
        response_format="structured",
        client=SimpleNamespace(responses=responses),
    )

    response = await client.generate(
        GenerationRequest(
            prompt="Create a document map.",
            prompt_name="stuffing/map",
            metadata={"task": "document_map"},
            response_model=StructuredDocumentMap,
        )
    )
    document_map = DocumentMap.model_validate_json(response.content)

    assert document_map.document_id == "doc_alpha"
    assert document_map.entries[0].attributes == {"priority": "high"}
    assert document_map.facets == {"topic": "contracts"}
    assert response.metadata["response_format"] == "structured"
    assert responses.parse_kwargs["model"] == "gpt-5-mini-doc-map-generator"
    assert responses.parse_kwargs["text_format"] is StructuredDocumentMap
    assert responses.parse_kwargs["max_output_tokens"] == 256
    assert "temperature" not in responses.parse_kwargs
    assert "text" not in responses.parse_kwargs


async def test_structured_generation_requires_response_model() -> None:
    client = OpenAICompatibleTextGenerationClient(
        base_url="https://example.openai.azure.com/openai/v1/",
        deployment="gpt-5-mini-doc-map-generator",
        api="responses",
        api_key="test-key",
        response_format="structured",
        client=SimpleNamespace(responses=_FakeResponses()),
    )

    with pytest.raises(RuntimeError, match="response_model"):
        await client.generate(
            GenerationRequest(
                prompt="Create a document map.",
                prompt_name="stuffing/map",
            )
        )


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="chatcmpl-test",
            model="doc-map",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='{"ok": true}'),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        )


class _FakeResponses:
    def __init__(self) -> None:
        self.kwargs = {}
        self.parse_kwargs = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="resp-test",
            model="gpt-5-mini-doc-map-generator",
            status="completed",
            error=None,
            incomplete_details=None,
            output_text='{"ok": true}',
            usage=SimpleNamespace(input_tokens=11, output_tokens=5),
        )

    def parse(self, **kwargs):
        self.parse_kwargs = kwargs
        return SimpleNamespace(
            id="resp-test",
            model="gpt-5-mini-doc-map-generator",
            status="completed",
            error=None,
            incomplete_details=None,
            output_parsed=StructuredDocumentMap(
                document_id="doc_alpha",
                overview="Alpha contract renewal approval.",
                entries=[
                    StructuredMapEntry(
                        id="alpha_approval",
                        kind="decision",
                        label="Board approval",
                        summary="The board approved the Alpha contract renewal.",
                        source_references=[
                            StructuredSourceReference(
                                document_id="doc_alpha",
                                segment_ids=["alpha_s1"],
                            )
                        ],
                        children=[],
                        attributes=[StructuredMapAttribute(key="priority", value="high")],
                    )
                ],
                facets=[StructuredMapAttribute(key="topic", value="contracts")],
                construction_method="stuffing",
            ),
            usage=SimpleNamespace(input_tokens=11, output_tokens=5),
        )
