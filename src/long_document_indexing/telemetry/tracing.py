from __future__ import annotations

import hashlib

from long_document_indexing.domain.runs import RunContext


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    """Create a stable short identifier from benchmark-controlled values."""

    seed = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:length]}"


def stable_query_run_id(
    experiment_id: str,
    system_id: str,
    item_id: str,
    repetition: int,
) -> str:
    return stable_id("run", experiment_id, system_id, item_id, repetition)


def stable_index_run_id(experiment_id: str, system_id: str, corpus_id: str) -> str:
    return stable_id("run", experiment_id, system_id, corpus_id, "index")


def stable_trace_id(context: RunContext, workflow_name: str) -> str:
    return stable_id(
        "trace",
        workflow_name,
        context.experiment_id,
        context.run_id,
        context.system_id,
        context.corpus_id,
        context.item_id or "",
        context.repetition,
    )
