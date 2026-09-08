Merge child maps into one parent map for hierarchical retrieval.

Document: ${document_id}
Title: ${title}
Construction method: ${construction_method}
Partial map count: ${partial_map_count}

Partial maps:
${partial_maps}

Return a single document map with source references preserved from the child maps. The result should summarize the child maps while keeping enough structure for query routing.

Output discipline:
- Use at most 4 root entries.
- Use an empty `children` list unless a child entry is essential; never use more than 1 child per root entry.
- Use at most 2 source references per entry and preserve only references that appear in the child maps.
- Keep `overview`, `label`, and each `summary` short; do not quote long passages.
