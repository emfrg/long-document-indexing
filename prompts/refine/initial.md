Create the initial retrieval-oriented document map for `${document_id}` from segment `${segment_id}`.

Title: `${title}`
Construction method: `${construction_method}`

Use the normalized DocumentMap schema. Preserve the source segment identifier.
Keep the map compact: prefer 1-2 useful entries, concise summaries, and source references instead of copied passages.

Output discipline:
- Use exactly 1 root entry unless the segment has two distinct answer-critical issues; never use more than 2.
- Use an empty `children` list for every entry.
- Use 1 source reference per entry and cite only the segment identifier from this prompt.
- Keep `overview`, `label`, and each `summary` short; do not quote long passages.

Segment:

`${segment_text}`
