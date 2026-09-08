Update the retrieval-oriented document map for `${document_id}` using segment `${segment_id}`.

Title: `${title}`
Construction method: `${construction_method}`

Keep useful existing entries, add important new information, and preserve source segment identifiers.

Use the normalized DocumentMap schema.
Keep the updated map compact: merge overlapping entries, prefer 1-4 useful entries, concise summaries, and source references instead of copied passages.

Output discipline:
- Use at most 4 root entries.
- Use an empty `children` list unless a child entry is essential; never use more than 1 child per root entry.
- Use at most 2 source references per entry and cite only existing references or the new segment identifier.
- Keep `overview`, `label`, and each `summary` short; do not quote long passages.

Existing map:

`${existing_map}`

New segment:

`${segment_text}`
