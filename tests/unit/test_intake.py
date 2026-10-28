"""Intake spec: fixed JSON schema the n8n LLM node fills; validation tells n8n what to re-ask."""

from __future__ import annotations

import json

from store_support.intake.schema import ISSUE_CATEGORIES, IssueSpec, missing_fields, validate_spec


def test_schema_is_fixed_and_exported_for_n8n() -> None:
    schema = IssueSpec.model_json_schema()
    assert set(schema["required"]) == {"order_id", "category", "description", "desired_outcome"}
    assert schema["properties"]["category"]["enum"] == list(ISSUE_CATEGORIES)
    json.dumps(schema)  # serializable


def test_missing_fields_lists_what_to_reask() -> None:
    assert missing_fields({"order_id": 1042}) == ["category", "description", "desired_outcome"]


def test_null_and_empty_count_as_missing() -> None:
    assert missing_fields({"order_id": 1042, "category": None, "description": "", "desired_outcome": "refund"}) == [
        "category", "description"]


def test_valid_spec_round_trip() -> None:
    ok, spec, errors = validate_spec({"order_id": "1042", "category": "damaged",
                                      "description": "steering wheel cover arrived damaged", "desired_outcome": "refund"})
    assert ok and not errors
    assert spec is not None and spec.order_id == 1042 and spec.category == "damaged"


def test_invalid_category_reports_field() -> None:
    ok, _, errors = validate_spec({"order_id": 1042, "category": "angry", "description": "x" * 10, "desired_outcome": "refund"})
    assert not ok and errors[0]["field"] == "category"


def test_spec_strips_pii_from_description() -> None:
    _, spec, _ = validate_spec({"order_id": 1042, "category": "damaged",
                                "description": "it broke, call me at 312-555-0147", "desired_outcome": "refund"})
    assert spec is not None and "555-0147" not in spec.description
