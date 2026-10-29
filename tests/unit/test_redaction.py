"""PII never reaches logs or traces. Presidio when installed, regex fallback otherwise."""

from __future__ import annotations

import logging

import pytest
import structlog

from store_support.governance.redaction import redact_event, redact_text


def test_email_redacted() -> None:
    assert "jane@example.com" not in redact_text("contact jane@example.com please")


def test_phone_redacted() -> None:
    out = redact_text("call me at 312-555-0147 or (312) 555-0147")
    assert "555-0147" not in out


def test_credit_card_redacted() -> None:
    assert "4111 1111 1111 1111" not in redact_text("card 4111 1111 1111 1111")


def test_order_ids_are_not_pii_and_survive() -> None:
    assert "1042" in redact_text("order 1042 arrived damaged")


def test_redaction_marker_names_entity_type() -> None:
    out = redact_text("email jane@example.com")
    assert "<EMAIL" in out.upper()


def test_structlog_processor_redacts_nested_values() -> None:
    event = {"event": "x", "detail": {"msg": "mail bob@x.io", "items": ["555-123-4567"]}}
    out = redact_event(None, "info", event)
    assert "bob@x.io" not in str(out)
    assert "555-123-4567" not in str(out)


def test_configured_logger_redacts(capsys) -> None:
    from store_support.governance.logging_setup import configure_logging

    configure_logging(level=logging.INFO, json=True)
    try:
        structlog.get_logger("t").info("hello", who="sam@corp.com")
        captured = capsys.readouterr()
        assert "sam@corp.com" not in captured.out + captured.err
    finally:
        structlog.reset_defaults()


# ---------- business identifiers are not PII (regression: Presidio tagged 1042 as DATE_TIME) ----------


@pytest.mark.parametrize(
    "text",
    [
        "order 1042 arrived damaged",
        "SKU-001 is out of stock",
        "ticket TCK-00012 opened, refund RFD-00002, appointment APT-00003",
        "refund 2499 cents on order 1042 for SKU-001",
    ],
)
def test_business_identifiers_survive_redaction(text: str) -> None:
    assert redact_text(text) == text


def test_person_and_location_redacted_when_presidio_available() -> None:
    from store_support.governance.redaction import _presidio

    if _presidio() is None:
        pytest.skip("presidio not installed")
    out = redact_text("Jane Doe in Chicago ordered SKU-001, order 1042")
    assert "Jane" not in out and "Chicago" not in out
    assert "SKU-001" in out and "1042" in out


# ---------- structured audit keys are identifiers, not free text (regression from live run) ----------


def test_structured_keys_are_not_redacted() -> None:
    event = {
        "event": "tool_call",
        "agent": "resolver-agent@1.0.0",
        "scopes": ["customer:read_own", "manager:refund"],
        "tool": "lookup_order",
        "tenant": "store-a",
        "subject": "u_aa223ddc7062",
        "detail": "contact jane@example.com",
    }
    out = redact_event(None, "info", dict(event))
    assert out["agent"] == "resolver-agent@1.0.0"
    assert out["scopes"] == ["customer:read_own", "manager:refund"]
    assert out["tool"] == "lookup_order"
    assert "jane@example.com" not in out["detail"]


def test_agent_version_string_is_not_an_email() -> None:
    assert redact_text("resolver-agent@1.0.0 called lookup_order") == "resolver-agent@1.0.0 called lookup_order"


def test_warm_up_is_idempotent_and_fast_after_first_call() -> None:
    import time

    from store_support.governance.redaction import warm_up

    warm_up()
    t = time.perf_counter()
    warm_up()
    assert time.perf_counter() - t < 0.5
