from __future__ import annotations

from long_document_indexing.domain.maps import DocumentMap
from long_document_indexing.models.base import GenerationRequest
from long_document_indexing.models.fake import FakeTextGenerationClient


async def test_fake_model_generates_document_map_json() -> None:
    client = FakeTextGenerationClient()

    response = await client.generate(
        GenerationRequest(
            prompt="Map this segment.",
            prompt_name="test",
            metadata={
                "task": "document_map",
                "strategy": "stuffing",
                "document_id": "doc_alpha",
                "segments": [
                    {
                        "id": "alpha_s1",
                        "document_id": "doc_alpha",
                        "order": 1,
                        "text": "Alpha renewal approval from the board.",
                        "metadata": {},
                    }
                ],
            },
        )
    )

    document_map = DocumentMap.model_validate_json(response.content)

    assert document_map.document_id == "doc_alpha"
    assert document_map.construction_method == "stuffing"
    assert document_map.entries[0].source_references[0].segment_ids == ["alpha_s1"]
    assert response.usage.model_calls == 1
