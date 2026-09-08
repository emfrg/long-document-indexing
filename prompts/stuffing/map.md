Create a retrieval-oriented document map for `${document_id}`.

Title: `${title}`
Segment count: `${segment_count}`
Construction method: `${construction_method}`

Use the normalized DocumentMap schema. Every entry must cite source segment identifiers.
Keep the map compact: prefer 1-3 useful entries, concise summaries, and source references instead of copied passages.

Output discipline:
- Use at most 3 root entries.
- Use an empty `children` list unless a child entry is essential; never use more than 1 child per root entry.
- Use at most 2 source references per entry and cite only segment identifiers from this prompt.
- Keep `overview`, `label`, and each `summary` short; do not quote long passages.

Segments:

`${segments}`
