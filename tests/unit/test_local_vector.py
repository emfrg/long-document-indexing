from __future__ import annotations

from long_document_indexing.domain.corpus import Corpus, Document, Segment
from long_document_indexing.retrieval.local_vector import LocalVectorBackend


async def test_local_vector_returns_relevant_segment_first(tmp_path) -> None:
    corpus = Corpus(
        id="corpus_1",
        documents=[
            Document(
                id="doc_alpha",
                corpus_id="corpus_1",
                segments=[
                    Segment(
                        id="alpha_s1",
                        document_id="doc_alpha",
                        order=1,
                        text="Alpha renewal approval from the board.",
                    )
                ],
            ),
            Document(
                id="doc_beta",
                corpus_id="corpus_1",
                segments=[
                    Segment(
                        id="beta_s1",
                        document_id="doc_beta",
                        order=1,
                        text="Beta storage incident timeline.",
                    )
                ],
            ),
        ],
    )
    backend = LocalVectorBackend(tmp_path)

    index_id = await backend.index(corpus)
    results = await backend.search(
        index_id,
        "alpha board approval",
        document_ids=None,
        top_k=2,
    )

    assert results[0].document_id == "doc_alpha"
    assert results[0].segment_id == "alpha_s1"
