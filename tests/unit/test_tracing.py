"""One trace per request; a span per agent step and per tool call; no PII in attributes."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from store_support.access import AgentIdentity, Principal, RequestContext, Scope
from store_support.telemetry import tracing


@pytest.fixture
def exporter(monkeypatch) -> InMemorySpanExporter:
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("test"))
    return exp


def _ctx() -> RequestContext:
    return RequestContext(
        principal=Principal("cust-42", "store-a", frozenset({Scope.CUSTOMER_READ_OWN})),
        agent=AgentIdentity("knowledge-agent", "1.0.0"),
        request_id="req-abc",
    )


def test_request_span_wraps_step_and_tool_spans(exporter) -> None:
    with tracing.request_span(_ctx()):
        with tracing.step_span("retrieve", agent="knowledge-agent"):
            pass
        with tracing.tool_span("lookup_order", _ctx()):
            pass
    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert names == {"request", "step.retrieve", "tool.lookup_order"}
    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 1  # one trace per request


def test_span_attributes_separate_agent_and_user_and_hide_raw_subject(exporter) -> None:
    with tracing.tool_span("lookup_order", _ctx()):
        pass
    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs["agent.name"] == "knowledge-agent"
    assert attrs["user.subject"] != "cust-42"
    assert attrs["user.tenant"] == "store-a"
    assert attrs["request.id"] == "req-abc"


def test_step_span_records_exception_and_still_closes(exporter) -> None:
    with pytest.raises(RuntimeError), tracing.step_span("grade", agent="knowledge-agent"):
        raise RuntimeError("boom")
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code.name == "ERROR"


def test_attribute_values_are_redacted(exporter) -> None:
    with tracing.step_span("intake", agent="intake", attributes={"question": "mail me at joe@x.com"}):
        pass
    attrs = exporter.get_finished_spans()[0].attributes
    assert "joe@x.com" not in attrs["question"]
