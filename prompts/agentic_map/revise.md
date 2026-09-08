Revise an agentic document map after inspecting one more source segment.

Document: ${document_id}
Title: ${title}
Construction method: ${construction_method}
Segment id: ${segment_id}

Segment:
${segment_text}

Existing map:
${existing_map}

Return the revised document map. Add or update entries only when the inspected segment supports them, and preserve source references.

Output discipline:
- Use at most 4 root entries.
- Use an empty `children` list unless a child entry is essential; never use more than 1 child per root entry.
- Use at most 2 source references per entry and cite only existing references or the inspected segment identifier.
- Keep `overview`, `label`, and each `summary` short; do not quote long passages.
