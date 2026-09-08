Fill one outline node with source-grounded map entries.

Document: ${document_id}
Title: ${title}
Construction method: ${construction_method}
Phase: ${phase}
Outline node:
${outline_node}

Source segment ids: ${source_segment_ids}

Segments:
${segments}

Return a document map for this outline node. Preserve exact segment source references.

Output discipline:
- Use 1-3 root entries for this outline node.
- Use an empty `children` list unless a child entry is essential; never use more than 1 child per root entry.
- Use at most 2 source references per entry and cite only segment identifiers from this prompt.
- Keep `overview`, `label`, and each `summary` short; do not quote long passages.
