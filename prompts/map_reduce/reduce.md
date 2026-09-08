Reduce `${partial_map_count}` partial maps into one retrieval-oriented document map for `${document_id}`.

Title: `${title}`
Construction method: `${construction_method}`

Do not invent source references. Preserve the source references already present in the partial maps.

Use the normalized DocumentMap schema.
Keep the reduced map compact: merge overlapping entries, prefer 1-4 useful entries, concise summaries, and source references instead of copied passages.

Output discipline:
- Use at most 4 root entries.
- Use an empty `children` list unless a child entry is essential; never use more than 1 child per root entry.
- Use at most 2 source references per entry and preserve only references that appear in the partial maps.
- Keep `overview`, `label`, and each `summary` short; do not quote long passages.

Partial maps:

`${partial_maps}`
