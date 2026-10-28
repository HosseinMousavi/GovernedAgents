"""The intake spec. n8n's LLM node fills this exact schema; the agents service validates it
and returns the list of fields to re-ask. Free text is PII-redacted on the way in."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from store_support.governance.redaction import redact_text

ISSUE_CATEGORIES = ("damaged", "missing", "wrong_item", "defective", "other")
DESIRED_OUTCOMES = ("refund", "replacement", "repair", "information")
REQUIRED = ("order_id", "category", "description", "desired_outcome")


class IssueSpec(BaseModel):
    order_id: int = Field(ge=1, description="Order number from the receipt")
    category: Literal["damaged", "missing", "wrong_item", "defective", "other"]
    description: str = Field(min_length=5, max_length=1000, description="What happened, in the customer's words")
    desired_outcome: Literal["refund", "replacement", "repair", "information"]
    product_hint: str | None = Field(default=None, max_length=120, description="Product name if mentioned")

    @field_validator("description", "product_hint")
    @classmethod
    def _redact(cls, v: str | None) -> str | None:
        return redact_text(v) if v else v


def missing_fields(data: dict[str, Any]) -> list[str]:
    return [f for f in REQUIRED if data.get(f) in (None, "", [])]


def validate_spec(data: dict[str, Any]) -> tuple[bool, IssueSpec | None, list[dict[str, str]]]:
    try:
        return True, IssueSpec.model_validate(data), []
    except ValidationError as e:
        errors = [{"field": ".".join(str(p) for p in err["loc"]) or "?", "problem": err["msg"]} for err in e.errors()]
        return False, None, errors
