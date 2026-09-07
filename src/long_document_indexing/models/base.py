from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from long_document_indexing.domain.runs import UsageRecord


class GenerationRequest(BaseModel):
    """Provider-neutral model generation request."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    prompt: str
    prompt_name: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    response_model: type[BaseModel] | None = Field(default=None, exclude=True)


class GenerationResponse(BaseModel):
    """Provider-neutral model generation response."""

    model_config = ConfigDict(extra="forbid")

    content: str
    usage: UsageRecord = Field(default_factory=UsageRecord)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextGenerationClient(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text for a provider-neutral request."""
