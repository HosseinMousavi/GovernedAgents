"""Findings from the independent security review, locked in as tests."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from store_support.config import Settings, SettingsError
from store_support.mcp_server.audit import AuditLog
from store_support.telemetry import tracing


def test_strict_settings_reject_default_signing_key(monkeypatch) -> None:
    monkeypatch.delenv("TOKEN_SIGNING_KEY", raising=False)
    with pytest.raises(SettingsError, match="TOKEN_SIGNING_KEY"):
        Settings.from_env(strict=True)
    monkeypatch.setenv("TOKEN_SIGNING_KEY", "dev-only-change-me")
    with pytest.raises(SettingsError):
        Settings.from_env(strict=True)
    monkeypatch.setenv("TOKEN_SIGNING_KEY", "x" * 32)
    assert Settings.from_env(strict=True).token_signing_key == "x" * 32


def test_exception_text_in_spans_is_redacted_and_has_no_stack(monkeypatch) -> None:
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("t"))
    with pytest.raises(RuntimeError), tracing.step_span("x", agent="a"):
        raise RuntimeError("failed for bob@corp.com")
    span = exp.get_finished_spans()[0]
    assert span.events == ()  # no record_exception event (would carry stacktrace + raw message)
    assert "bob@corp.com" not in str(dict(span.attributes))
    assert span.attributes["error.type"] == "RuntimeError"


def test_audit_buffer_is_bounded() -> None:
    from store_support.access import AgentIdentity, Principal, RequestContext

    log = AuditLog(max_entries=3)
    ctx = RequestContext(Principal("s", "t", frozenset()), AgentIdentity("a", "1"), "r")
    for i in range(5):
        log.record(ctx, tool=f"t{i}", outcome="ok")
    assert [e["tool"] for e in log.entries] == ["t2", "t3", "t4"]


def test_knowledge_ask_respects_kill_switch(monkeypatch, tmp_path) -> None:
    from starlette.testclient import TestClient

    from store_support.access import Scope, issue_token
    from store_support.resolver.service import build_app

    flag = tmp_path / "KILL"
    monkeypatch.setenv("KILL_SWITCH_FILE", str(flag))
    monkeypatch.setenv("KILL_SWITCH", "false")
    settings = Settings.from_env()
    client = TestClient(build_app(settings))
    tok = issue_token(
        subject_id="c", tenant_id="store-a", scopes=[Scope.CUSTOMER_READ_OWN], key=settings.token_signing_key
    )
    flag.write_text("1")
    r = client.post("/knowledge/ask", json={"question": "x"}, headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 503
