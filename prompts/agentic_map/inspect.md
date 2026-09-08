Inspect the first source segment and start an agentic document map.

Document: ${document_id}
Title: ${title}
Construction method: ${construction_method}
Segment id: ${segment_id}

Segment:
${segment_text}

Existing map:
${existing_map}

Return the current best document map. Preserve source references and only include claims supported by the inspected segment.

Output discipline:
- Use exactly 1 root entry unless the segment has two distinct answer-critical issues; never use more than 2.
- Use an empty `children` list for every entry.
- Use 1 source reference per entry and cite only the inspected segment identifier.
- Keep `overview`, `label`, and each `summary` short; do not quote long passages.
