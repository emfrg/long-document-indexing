from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from long_document_indexing.models.base import EmbeddingClient, TextGenerationClient
from long_document_indexing.prompts import PromptLoader
from long_document_indexing.retrieval.base import RetrievalBackend
from long_document_indexing.storage.artifacts import ArtifactStore
from long_document_indexing.telemetry.usage import UsageLedger
from long_document_indexing.workflows.execution import WorkflowRunner

AnswerMode = Literal["extractive", "generated"]


@dataclass(frozen=True)
class Services:
    """Runtime dependencies passed into systems and runners."""

    artifact_store: ArtifactStore
    retrieval_backend: RetrievalBackend
    workflow_runner: WorkflowRunner
    usage_ledger: UsageLedger
    prompt_loader: PromptLoader
    answering_mode: AnswerMode = "extractive"
    generator_client: TextGenerationClient | None = None
    router_client: TextGenerationClient | None = None
    answer_client: TextGenerationClient | None = None
    embedding_client: EmbeddingClient | None = None
    judge_client: object | None = None
    map_generation_signature: str = "unspecified-map-generation"
    query_config_signature: str = "unspecified-query-config"
    index_config_signatures: dict[str, str] | None = None
    resume_checkpoints: bool = False
    progress: Callable[[str], None] | None = None

    def emit_progress(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def index_config_signature(self, system_id: str) -> str:
        if self.index_config_signatures is None:
            return "unspecified-index-config"
        return self.index_config_signatures.get(system_id, "unspecified-index-config")
