"""PII never reaches logs or traces. Presidio when installed, regex fallback otherwise."""

from __future__ import annotations

import logging

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
    structlog.get_logger("t").info("hello", who="sam@corp.com")
    captured = capsys.readouterr()
    assert "sam@corp.com" not in captured.out + captured.err
