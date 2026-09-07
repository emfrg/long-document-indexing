from __future__ import annotations

from long_document_indexing.text import lexical_similarity


def test_lexical_similarity_uses_content_words_for_short_queries() -> None:
    query = "When did the beta incident start and end?"
    unrelated_map = (
        "The board approved the Alpha contract renewal after procurement confirmed "
        "vendor performance."
    )
    relevant_map = "Storage outage incident on April 8, 2024: started 09:15, restored 11:40."

    assert lexical_similarity(query, relevant_map) > lexical_similarity(query, unrelated_map)
