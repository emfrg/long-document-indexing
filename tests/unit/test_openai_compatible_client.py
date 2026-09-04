from __future__ import annotations

from types import SimpleNamespace

from long_document_indexing.models.base import GenerationRequest
from long_document_indexing.models.openai_compatible import OpenAICompatibleTextGenerationClient


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
