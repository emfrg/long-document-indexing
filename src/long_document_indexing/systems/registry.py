from __future__ import annotations

from long_document_indexing.config import SystemConfig, normalize_system_id
from long_document_indexing.systems.agentic_map import AgenticMapSystem
from long_document_indexing.systems.base import RagSystem
from long_document_indexing.systems.flat_vector import FlatVectorSystem
from long_document_indexing.systems.hierarchical_map import HierarchicalMapSystem
from long_document_indexing.systems.map_reduce import MapReduceSystem
from long_document_indexing.systems.outline_then_fill import OutlineThenFillSystem
from long_document_indexing.systems.refine import RefineSystem
from long_document_indexing.systems.stuffing import StuffingSystem


def create_system(system_id: str, config: SystemConfig | None = None) -> RagSystem:
    normalized = normalize_system_id(system_id)
    config = config or SystemConfig(id=normalized, indexing_strategy=normalized)
    if normalized == FlatVectorSystem.id:
        return FlatVectorSystem()
    if normalized == StuffingSystem.id:
        return StuffingSystem()
    if normalized == MapReduceSystem.id:
        return MapReduceSystem(reduce_fan_in=config.reduce_fan_in)
    if normalized == RefineSystem.id:
        return RefineSystem()
    if normalized == HierarchicalMapSystem.id:
        return HierarchicalMapSystem(
            branching_factor=config.hierarchy_branching_factor,
            max_levels=config.hierarchy_max_levels,
        )
    if normalized == OutlineThenFillSystem.id:
        return OutlineThenFillSystem(
            outline_depth=config.outline_depth,
            outline_max_nodes=config.outline_max_nodes,
        )
    if normalized == AgenticMapSystem.id:
        return AgenticMapSystem(
            max_steps=config.agent_max_steps,
            target_coverage=config.agent_target_coverage,
        )
    raise ValueError(f"unknown or unimplemented system: {system_id}")
