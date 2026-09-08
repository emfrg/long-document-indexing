Plan a retrieval-oriented outline before filling the document map.

Document: ${document_id}
Title: ${title}
Construction method: ${construction_method}
Outline depth: ${outline_depth}
Maximum outline nodes: ${outline_max_nodes}
Segment count: ${segment_count}

Segments:
${segments}

Return a document map whose entries act as the outline nodes to fill. Keep the outline compact and source-grounded.

Output discipline:
- Use at most `${outline_max_nodes}` root/child entries total, and prefer fewer when possible.
- Use short outline labels and summaries; do not copy source passages.
- Use source references only when a source segment directly motivates an outline node.
- Keep `children` shallow and only when needed for the requested outline depth.
