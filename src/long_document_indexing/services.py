from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from long_document_indexing.models.base import TextGenerationClient
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
    embedding_client: object | None = None
    judge_client: object | None = None
