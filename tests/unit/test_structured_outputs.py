from __future__ import annotations

from long_document_indexing.models.structured_outputs import StructuredDocumentMap


def test_structured_document_map_schema_uses_closed_objects() -> None:
    schema = StructuredDocumentMap.model_json_schema()

    assert _object_paths_with_open_properties(schema) == []


def _object_paths_with_open_properties(value: object, path: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        if value.get("type") == "object" and value.get("additionalProperties") is not False:
            paths.append(path)
        for key, item in value.items():
            paths.extend(_object_paths_with_open_properties(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_object_paths_with_open_properties(item, f"{path}[{index}]"))
    return paths
