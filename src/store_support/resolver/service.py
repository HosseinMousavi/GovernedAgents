"""Agents HTTP service - what n8n talks to.

GET  /healthz
GET  /intake/schema            IssueSpec JSON schema (n8n's LLM node fills this)
POST /intake/validate          {spec} -> {ok, missing, errors}
POST /resolve                  Authorization: Bearer <customer token>; body {spec} -> Resolution
POST /knowledge/ask            Authorization: Bearer <token>; body {question} -> KnowledgeResult
GET  /manifests                loaded agent manifests
POST /tokens/dev               DEV ONLY: mint a scoped token (disabled unless DEV_TOKENS=true)
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from store_support.access import AgentIdentity, RequestContext, TokenError, issue_token, parse_token
from store_support.config import Settings
from store_support.governance.kill_switch import KillSwitch
from store_support.governance.logging_setup import configure_logging
from store_support.governance.manifest import load_all, load_manifest
from store_support.intake.schema import IssueSpec, missing_fields, validate_spec
from store_support.knowledge.factory import KnowledgeService
from store_support.resolver.agent import ResolverAgent
from store_support.resolver.mcp_client import McpToolClient
from store_support.telemetry.tracing import configure_tracing, request_span

ROOT = Path(__file__).resolve().parents[3]


def _bearer(request: Request, key: str) -> tuple[str, Any]:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise TokenError("missing bearer token")
    token = auth[7:].strip()
    return token, parse_token(token, key=key)


def build_app(settings: Settings) -> Starlette:
    manifests = {m.name: m for m in load_all(ROOT / "agents")}
    resolver_manifest = load_manifest(ROOT / "agents" / "resolver-agent" / "manifest.yaml")
    kill_switch = KillSwitch()
    _tokens: dict[str, str] = {}  # request_id -> raw token, so the MCP client can forward it

    def token_for(ctx: RequestContext) -> str:
        return _tokens[ctx.request_id]

    tools = McpToolClient(settings.mcp_server_url, token_for=token_for)

    async def healthz(_: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "kill_switch": kill_switch.engaged})

    async def intake_schema(_: Request) -> JSONResponse:
        return JSONResponse(IssueSpec.model_json_schema())

    async def intake_validate(request: Request) -> JSONResponse:
        body = await request.json()
        data = body.get("spec", body)
        missing = missing_fields(data)
        if missing:
            return JSONResponse({"ok": False, "missing": missing, "errors": []})
        ok, _spec, errors = validate_spec(data)
        return JSONResponse({"ok": ok, "missing": [], "errors": errors})

    async def resolve(request: Request) -> JSONResponse:
        try:
            raw, principal = _bearer(request, settings.token_signing_key)
        except TokenError as e:
            return JSONResponse({"error": str(e)}, status_code=401)
        if kill_switch.engaged:
            return JSONResponse({"error": "kill switch engaged"}, status_code=503)
        body = await request.json()
        ok, spec, errors = validate_spec(body.get("spec", body))
        if not ok or spec is None:
            return JSONResponse({"error": "invalid spec", "errors": errors}, status_code=422)
        ctx = RequestContext(
            principal=principal,
            request_id=uuid.uuid4().hex,
            agent=AgentIdentity(resolver_manifest.name, resolver_manifest.version),
        )
        _tokens[ctx.request_id] = raw
        flavor = os.environ.get("KNOWLEDGE_FLAVOR", "langgraph")
        try:
            with KnowledgeService(settings, principal.tenant_id, flavor=flavor) as knowledge:  # type: ignore[arg-type]
                agent = ResolverAgent(manifest=resolver_manifest, knowledge=_SyncKnowledge(knowledge), tools=tools)
                result = await run_in_threadpool(agent.resolve, ctx, spec)
        finally:
            _tokens.pop(ctx.request_id, None)
        return JSONResponse({"request_id": ctx.request_id, **result.to_dict()})

    async def knowledge_ask(request: Request) -> JSONResponse:
        try:
            _raw, principal = _bearer(request, settings.token_signing_key)
        except TokenError as e:
            return JSONResponse({"error": str(e)}, status_code=401)
        if kill_switch.engaged:
            return JSONResponse({"error": "kill switch engaged"}, status_code=503)
        body = await request.json()
        ctx = RequestContext(
            principal=principal, request_id=uuid.uuid4().hex, agent=AgentIdentity("knowledge-agent", "1.0.0")
        )
        with request_span(ctx), KnowledgeService(settings, principal.tenant_id) as svc:
            res = await svc.ask(str(body["question"]))
        return JSONResponse(
            {
                "outcome": res.outcome.value,
                "answer": res.answer,
                "reason": res.reason,
                "citations": [f"{c.source}#{c.section}" for c in res.citations],
                "retries": res.retries,
                "scores": res.scores,
            }
        )

    async def list_manifests(_: Request) -> JSONResponse:
        return JSONResponse({k: m.model_dump() for k, m in manifests.items()})

    async def dev_token(request: Request) -> JSONResponse:
        if os.environ.get("DEV_TOKENS", "").lower() != "true":
            return JSONResponse({"error": "disabled"}, status_code=404)
        b = await request.json()
        tok = issue_token(
            subject_id=b["subject_id"], tenant_id=b["tenant_id"], scopes=b["scopes"], key=settings.token_signing_key
        )
        return JSONResponse({"token": tok})

    return Starlette(
        routes=[
            Route("/healthz", healthz),
            Route("/intake/schema", intake_schema),
            Route("/intake/validate", intake_validate, methods=["POST"]),
            Route("/resolve", resolve, methods=["POST"]),
            Route("/knowledge/ask", knowledge_ask, methods=["POST"]),
            Route("/manifests", list_manifests),
            Route("/tokens/dev", dev_token, methods=["POST"]),
        ]
    )


class _SyncKnowledge:
    """Adapts the async KnowledgeService to the resolver's sync `run` (runs inside a threadpool)."""

    def __init__(self, svc: KnowledgeService) -> None:
        self._svc = svc

    def run(self, question: str) -> Any:
        import asyncio

        return asyncio.run(self._svc.ask(question))


def main() -> None:
    configure_logging(level=logging.INFO)
    settings = Settings.from_env(strict=True)
    configure_tracing(service_name=settings.otel_service_name, endpoint=settings.otel_endpoint)
    uvicorn.run(build_app(settings), host="0.0.0.0", port=8080)  # noqa: S104


if __name__ == "__main__":
    main()
