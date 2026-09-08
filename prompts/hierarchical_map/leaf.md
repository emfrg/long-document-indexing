Create a source-grounded leaf document map for one segment group.

Document: ${document_id}
Title: ${title}
Construction method: ${construction_method}
Hierarchy level: ${hierarchy_level}
Group index: ${group_index}
Source segment ids: ${source_segment_ids}

Segments:
${segments}

Return a document map that preserves segment-level source references. Use concise labels and summaries that help route future questions to the right document.

Output discipline:
- Use 1-2 root entries for this segment group.
- Use an empty `children` list unless a child entry is essential; never use more than 1 child per root entry.
- Use at most 2 source references per entry and cite only segment identifiers from this prompt.
- Keep `overview`, `label`, and each `summary` short; do not quote long passages.
