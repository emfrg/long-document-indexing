from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from long_document_indexing.domain.benchmark import DatasetCapabilities

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")
_ENV_NAME_PATTERN = re.compile(r"[A-Z0-9_]+")


class ExperimentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    mode: str = "controlled_ablation"
    seed: int = 42
    query_repetitions: int = 1

    @field_validator("id", "mode")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter: str
    corpus_path: Path | None = None
    question_set: Path | None = None
    revision: str | None = None
    split: str | None = None
    case_manifest: Path | None = None
    capabilities: DatasetCapabilities = Field(default_factory=DatasetCapabilities)
    options: dict[str, Any] = Field(default_factory=dict)

    def resolve_paths(self, project_root: Path) -> DatasetConfig:
        updates: dict[str, Path] = {}
        for field_name in ("corpus_path", "question_set", "case_manifest"):
            value = getattr(self, field_name)
            if value is not None and not value.is_absolute():
                updates[field_name] = (project_root / value).resolve()
        return self.model_copy(update=updates)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generator_provider: Literal["fake", "foundry", "openai_compatible"] = "fake"
    generator_api: Literal["chat_completions", "responses"] = "chat_completions"
    generator_deployment: str | None = None
    generator_base_url: str | None = None
    generator_api_key_env: str = "AZURE_INFERENCE_CREDENTIAL"
    generator_auth_mode: Literal["api_key", "azure_default_credential"] = "api_key"
    generator_azure_scope: str = "https://ai.azure.com/.default"
    generator_temperature: float = 0.0
    generator_max_output_tokens: int | None = None
    generator_timeout_seconds: float = 60.0
    generator_response_format: Literal["structured", "json_object", "text"] = "json_object"
    judge_deployment: str | None = None
    embedding_deployment: str | None = None

    @field_validator("generator_temperature")
    @classmethod
    def _temperature_in_supported_range(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("generator_temperature must be between 0 and 2")
        return value

    @field_validator("generator_max_output_tokens")
    @classmethod
    def _max_output_tokens_positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("generator_max_output_tokens must be positive when set")
        return value

    @field_validator("generator_timeout_seconds")
    @classmethod
    def _timeout_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("generator_timeout_seconds must be positive")
        return value


class SharedPipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_tokens: int = 6000
    segment_overlap_tokens: int = 250
    selected_documents: int = 3
    retrieved_segments: int = 12


class WorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runner: Literal["local", "maf"] = "local"


class AnsweringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["extractive", "generated"] = "extractive"


class RunControlConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resume: bool = False
    force: bool = False
    max_model_calls: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_estimated_cost: float | None = None

    @field_validator(
        "max_model_calls",
        "max_input_tokens",
        "max_output_tokens",
        "max_total_tokens",
    )
    @classmethod
    def _integer_budget_must_not_be_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("budget limits must not be negative")
        return value

    @field_validator("max_estimated_cost")
    @classmethod
    def _cost_budget_must_not_be_negative(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("budget limits must not be negative")
        return value

    @model_validator(mode="after")
    def _resume_and_force_are_mutually_exclusive(self) -> RunControlConfig:
        if self.resume and self.force:
            raise ValueError("run_control.resume and run_control.force cannot both be true")
        return self

    @property
    def has_budget_limits(self) -> bool:
        return any(
            value is not None
            for value in (
                self.max_model_calls,
                self.max_input_tokens,
                self.max_output_tokens,
                self.max_total_tokens,
                self.max_estimated_cost,
            )
        )


class FoundryEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    dataset_path: Path = Path("evaluations/foundry/dataset.jsonl")
    manifest_path: Path = Path("evaluations/foundry/manifest.json")
    result_path: Path = Path("evaluations/foundry/result.json")
    azure_ai_project: str | None = None
    evaluation_name: str | None = None
    evaluation_level: Literal["turn"] = "turn"
    evaluators: list[str] = Field(
        default_factory=lambda: [
            "groundedness",
            "relevance",
            "retrieval",
            "document_retrieval",
        ]
    )
    managed_evaluators: list[str] = Field(default_factory=lambda: ["f1", "rouge"])
    fail_on_evaluator_errors: bool = False
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("dataset_path", "manifest_path", "result_path")
    @classmethod
    def _must_be_artifact_relative_path(cls, value: Path) -> Path:
        if value.is_absolute() or any(part == ".." for part in value.parts):
            raise ValueError("path must be artifact-relative")
        if str(value).strip() in {"", "."}:
            raise ValueError("path must not be blank")
        return value

    @field_validator("azure_ai_project", "evaluation_name")
    @classmethod
    def _optional_strings_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("evaluators", "managed_evaluators")
    @classmethod
    def _evaluators_must_be_unique_and_non_blank(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("evaluators must not contain blank values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("evaluators must not contain duplicates")
        return normalized

    @field_validator("tags")
    @classmethod
    def _tags_must_not_be_blank(cls, value: dict[str, str]) -> dict[str, str]:
        for key, tag_value in value.items():
            if not key.strip() or not tag_value.strip():
                raise ValueError("tags must not contain blank keys or values")
        return value


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local: list[str] = Field(default_factory=list)
    foundry: FoundryEvaluationConfig = Field(default_factory=FoundryEvaluationConfig)


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifacts_dir: Path = Path("artifacts")

    def resolve_paths(self, project_root: Path) -> StorageConfig:
        if self.artifacts_dir.is_absolute():
            return self
        return self.model_copy(
            update={"artifacts_dir": (project_root / self.artifacts_dir).resolve()}
        )


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentMetadata
    dataset: DatasetConfig
    models: ModelConfig = Field(default_factory=ModelConfig)
    shared_pipeline: SharedPipelineConfig = Field(default_factory=SharedPipelineConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    answering: AnsweringConfig = Field(default_factory=AnsweringConfig)
    run_control: RunControlConfig = Field(default_factory=RunControlConfig)
    systems: list[str]
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @model_validator(mode="after")
    def _validate_systems(self) -> ExperimentConfig:
        if not self.systems:
            raise ValueError("at least one system must be configured")
        if len(self.systems) != len(set(self.systems)):
            raise ValueError("systems list contains duplicates")
        return self

    def resolve_paths(self, project_root: Path) -> ExperimentConfig:
        return self.model_copy(
            update={
                "dataset": self.dataset.resolve_paths(project_root),
                "storage": self.storage.resolve_paths(project_root),
            }
        )


def load_experiment_config(path: Path, project_root: Path | None = None) -> ExperimentConfig:
    project_root = project_root or Path.cwd()
    _load_local_env(project_root / ".env")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"empty config file: {path}")

    expanded = _expand_env(raw)
    return ExperimentConfig.model_validate(expanded).resolve_paths(project_root)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_string(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _expand_env_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        return match.group(0)

    return _ENV_PATTERN.sub(replace, value)


def _load_local_env(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not _ENV_NAME_PATTERN.fullmatch(name):
            continue

        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name, value)
