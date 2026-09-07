from __future__ import annotations

import time

from long_document_indexing.answering import answer_from_retrieved_evidence, query_usage
from long_document_indexing.config import SharedPipelineConfig
from long_document_indexing.domain.benchmark import BenchmarkItem
from long_document_indexing.domain.corpus import Corpus
from long_document_indexing.domain.maps import IndexArtifact
from long_document_indexing.domain.runs import RagRunRecord
from long_document_indexing.services import Services
from long_document_indexing.telemetry.tracing import stable_query_run_id


class FlatVectorSystem:
    """Baseline that indexes raw segments and answers from top-k retrieved evidence."""

    id = "flat_vector"

    async def build_index(
        self,
        corpus: Corpus,
        services: Services,
        pipeline: SharedPipelineConfig,
    ) -> IndexArtifact:
        del pipeline
        index_id = await services.retrieval_backend.index(corpus)
        artifact_path = services.retrieval_backend.artifact_path(index_id)
        return IndexArtifact(
            id=index_id,
            system_id=self.id,
            corpus_id=corpus.id,
            artifact_path=str(artifact_path) if artifact_path is not None else index_id,
            build_metadata={"backend": services.retrieval_backend.__class__.__name__},
        )

    async def run_query(
        self,
        item: BenchmarkItem,
        corpus: Corpus,
        index_artifact: IndexArtifact,
        services: Services,
        pipeline: SharedPipelineConfig,
        *,
        experiment_id: str,
        repetition: int,
    ) -> RagRunRecord:
        del corpus
        started = time.perf_counter()
        retrieved_items = await services.retrieval_backend.search(
            index_artifact.id,
            item.query,
            document_ids=None,
            top_k=pipeline.retrieved_segments,
        )
        selected_document_ids = _unique_document_ids(retrieved_items)[: pipeline.selected_documents]
        answer_result = await answer_from_retrieved_evidence(
            item=item,
            retrieved_items=retrieved_items[: pipeline.retrieved_segments],
            services=services,
            extractive_prefix="Local baseline answer from retrieved evidence",
        )
        duration_ms = (time.perf_counter() - started) * 1000.0
        return RagRunRecord(
            run_id=stable_query_run_id(experiment_id, self.id, item.id, repetition),
            experiment_id=experiment_id,
            system_id=self.id,
            corpus_id=item.corpus_id,
            item_id=item.id,
            repetition=repetition,
            selected_document_ids=selected_document_ids,
            retrieved_items=retrieved_items,
            answer=answer_result.answer,
            citations=answer_result.citations,
            usage=query_usage(
                answer_result.usage,
                tool_calls=1,
                duration_ms=duration_ms,
            ),
            status="succeeded",
        )


def _unique_document_ids(retrieved_items: list) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in retrieved_items:
        if item.document_id not in seen:
            ordered.append(item.document_id)
            seen.add(item.document_id)
    return ordered
