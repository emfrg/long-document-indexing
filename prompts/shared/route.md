Act as the document-routing agent for a long-document RAG system.

Question:
`${query}`

Select at most `${max_documents}` source documents that should be searched for
raw evidence. Inspect the complete structured document maps, including their
entries, summaries, relationships, and source references.

Requirements:
- Return document identifiers only from the supplied maps.
- Rank the selected identifiers from most to least useful.
- Select multiple documents when the question requires evidence from multiple sources.
- Do not answer the question.
- Explain which map entries or mapped issues motivated the routing decision.
- Record any evidence needs that the maps do not resolve.

Document maps:

`${document_maps}`
