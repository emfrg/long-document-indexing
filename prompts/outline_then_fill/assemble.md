Assemble filled outline-node maps into one final document map.

Document: ${document_id}
Title: ${title}
Construction method: ${construction_method}
Partial map count: ${partial_map_count}

Filled maps:
${partial_maps}

Return a single document map that keeps the outline-oriented structure and preserves all source references from the filled maps.

Output discipline:
- Use at most 4 root entries in the final map.
- Use an empty `children` list unless a child entry is essential; never use more than 1 child per root entry.
- Use at most 2 source references per entry and preserve only references that appear in the filled maps.
- Keep `overview`, `label`, and each `summary` short; do not quote long passages.
