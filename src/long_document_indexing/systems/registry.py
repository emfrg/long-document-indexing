from __future__ import annotations

from long_document_indexing.systems.base import RagSystem
from long_document_indexing.systems.flat_vector import FlatVectorSystem
from long_document_indexing.systems.map_reduce import MapReduceSystem
from long_document_indexing.systems.refine import RefineSystem
from long_document_indexing.systems.stuffing import StuffingSystem


def create_system(system_id: str) -> RagSystem:
    normalized = system_id.replace("-", "_")
    if normalized == FlatVectorSystem.id:
        return FlatVectorSystem()
    if normalized == StuffingSystem.id:
        return StuffingSystem()
    if normalized == MapReduceSystem.id:
        return MapReduceSystem()
    if normalized == RefineSystem.id:
        return RefineSystem()
    raise ValueError(f"unknown or unimplemented system: {system_id}")
