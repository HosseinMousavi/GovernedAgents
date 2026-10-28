"""OpenTelemetry tracing.

Local  : OTLP/gRPC -> Jaeger (docker-compose).
AWS    : OTLP/gRPC -> ADOT collector sidecar -> AWS X-Ray + CloudWatch (see infra/terraform).
The code is identical; only OTEL_EXPORTER_OTLP_ENDPOINT changes.

Span naming:  request  >  step.<name>  >  tool.<name>
Attribute values pass through the PII redactor; user subject is pseudonymized.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from store_support.access import RequestContext
from store_support.governance.redaction import redact_text
from store_support.mcp_server.audit import pseudonymize

_tracer = trace.get_tracer("store-support-agent")


def configure_tracing(*, service_name: str, endpoint: str) -> None:
    """Install a real provider. Call once at process start; no-op in unit tests."""
    global _tracer
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(provider)
    _tracer = provider.get_tracer("store-support-agent")


def _clean(attrs: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (attrs or {}).items():
        out[k] = redact_text(v) if isinstance(v, str) else v
    return out


def _identity_attrs(ctx: RequestContext) -> dict[str, Any]:
    return {
        "request.id": ctx.request_id,
        "agent.name": ctx.agent.name,
        "agent.version": ctx.agent.version,
        "user.subject": pseudonymize(ctx.principal.subject_id),
        "user.tenant": ctx.principal.tenant_id,
        "user.scopes": ",".join(sorted(s.value for s in ctx.principal.scopes)),
    }


@contextmanager
def _span(name: str, attrs: dict[str, Any]) -> Iterator[trace.Span]:
    with _tracer.start_as_current_span(
        name, attributes=attrs, record_exception=False, set_status_on_exception=False
    ) as span:
        try:
            yield span
        except Exception as e:
            # Deliberately NOT span.record_exception(): it attaches the raw message and a stack
            # trace, both of which can carry PII. Type + redacted message is enough to debug.
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", redact_text(str(e))[:200])
            span.set_status(Status(StatusCode.ERROR, type(e).__name__))
            raise


@contextmanager
def request_span(ctx: RequestContext) -> Iterator[trace.Span]:
    with _span("request", _identity_attrs(ctx)) as s:
        yield s


@contextmanager
def step_span(name: str, *, agent: str, attributes: Mapping[str, Any] | None = None) -> Iterator[trace.Span]:
    with _span(f"step.{name}", {"agent.name": agent, **_clean(attributes)}) as s:
        yield s


@contextmanager
def tool_span(name: str, ctx: RequestContext, attributes: Mapping[str, Any] | None = None) -> Iterator[trace.Span]:
    with _span(f"tool.{name}", {**_identity_attrs(ctx), "tool.name": name, **_clean(attributes)}) as s:
        yield s
