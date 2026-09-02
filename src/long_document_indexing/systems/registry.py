from __future__ import annotations

from long_document_indexing.systems.base import RagSystem
from long_document_indexing.systems.flat_vector import FlatVectorSystem


def create_system(system_id: str) -> RagSystem:
    normalized = system_id.replace("-", "_")
    if normalized == FlatVectorSystem.id:
        return FlatVectorSystem()
    raise ValueError(f"unknown or unimplemented system: {system_id}")
