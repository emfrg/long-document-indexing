from __future__ import annotations

from long_document_indexing.models.structured_outputs import (
    StructuredDocumentMap,
    StructuredGeneratedAnswer,
    StructuredMapAttribute,
    StructuredMapEntry,
    StructuredSourceReference,
)


def test_structured_document_map_schema_uses_closed_objects() -> None:
    schema = StructuredDocumentMap.model_json_schema()

    assert _object_paths_with_open_properties(schema) == []


def test_structured_generated_answer_schema_uses_closed_objects() -> None:
    schema = StructuredGeneratedAnswer.model_json_schema()

    assert _object_paths_with_open_properties(schema) == []


def test_structured_schemas_avoid_foundry_unsupported_type_constraints() -> None:
    unsupported_keywords = {
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "multipleOf",
        "patternProperties",
        "unevaluatedProperties",
        "propertyNames",
        "minProperties",
        "maxProperties",
        "unevaluatedItems",
        "contains",
        "minContains",
        "maxContains",
        "minItems",
        "maxItems",
        "uniqueItems",
    }

    for schema in (
        StructuredDocumentMap.model_json_schema(),
        StructuredGeneratedAnswer.model_json_schema(),
    ):
        assert _schema_keys(schema).isdisjoint(unsupported_keywords)


def test_structured_document_map_deduplicates_model_entry_ids() -> None:
    document_map = StructuredDocumentMap(
        document_id="doc_alpha",
        overview="Alpha overview.",
        entries=[
            _entry("same_id", "alpha_s1"),
            _entry("same_id", "alpha_s2"),
            StructuredMapEntry(
                id="parent",
                kind="section",
                label="Parent",
                summary="Parent summary.",
                source_references=[
                    StructuredSourceReference(document_id="doc_alpha", segment_ids=["alpha_s3"])
                ],
                children=[_entry("same_id", "alpha_s3")],
                attributes=[],
            ),
        ],
        facets=[StructuredMapAttribute(key="topic", value="alpha")],
        construction_method="agentic_map",
    ).to_document_map()

    entry_ids = [entry.id for root in document_map.entries for entry in root.walk()]
    assert entry_ids == ["same_id", "same_id_2", "parent", "same_id_3"]


def _entry(entry_id: str, segment_id: str) -> StructuredMapEntry:
    return StructuredMapEntry(
        id=entry_id,
        kind="segment_summary",
        label=segment_id,
        summary=f"Summary for {segment_id}.",
        source_references=[
            StructuredSourceReference(document_id="doc_alpha", segment_ids=[segment_id])
        ],
        children=[],
        attributes=[],
    )


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


def _schema_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(str(key) for key in value)
        for item in value.values():
            keys.update(_schema_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_schema_keys(item))
    return keys
