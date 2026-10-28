"""MCP server (Python MCP SDK, streamable HTTP).

Identity arrives on every HTTP request:
  Authorization: Bearer <scoped token>   -> Principal (user: who we act for)
  X-Agent-Name / X-Agent-Version         -> AgentIdentity (software: who is acting)

Tool functions are generated from the registry in tools.py, so the published JSON
schema is exactly the pydantic model each handler validates against.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from psycopg_pool import ConnectionPool

from store_support.access import AgentIdentity, RequestContext, TokenError, parse_token
from store_support.config import Settings
from store_support.governance.kill_switch import KillSwitch
from store_support.governance.logging_setup import configure_logging
from store_support.mcp_server.audit import AuditLog
from store_support.mcp_server.repo import Repository
from store_support.mcp_server.repo_postgres import PostgresRepository
from store_support.mcp_server.tools import TOOLS, ToolError, ToolSpec, execute_tool


class AuthError(Exception):
    pass


def context_from_headers(headers: dict[str, str] | None, *, key: str) -> RequestContext:
    headers = {k.lower(): v for k, v in (headers or {}).items()}
    auth = headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise AuthError("missing bearer token")
    try:
        principal = parse_token(auth[7:].strip(), key=key)
    except TokenError as e:
        raise AuthError(str(e)) from e
    agent = AgentIdentity(
        name=headers.get("x-agent-name", "unknown-agent"), version=headers.get("x-agent-version", "0")
    )
    return RequestContext(principal=principal, agent=agent, request_id=headers.get("x-request-id") or uuid.uuid4().hex)


def _make_tool_fn(spec: ToolSpec, run: Callable[[Context, str, dict[str, Any]], dict[str, Any]]) -> Callable[..., Any]:
    """Build `def <name>(ctx: Context, <field>: <type>, ...) -> dict` from the pydantic model."""
    params = [inspect.Parameter("ctx", inspect.Parameter.KEYWORD_ONLY, annotation=Context)]
    annotations: dict[str, Any] = {"ctx": Context, "return": dict[str, Any]}
    for fname, finfo in spec.model.model_fields.items():
        default = inspect.Parameter.empty if finfo.is_required() else finfo.default
        params.append(
            inspect.Parameter(fname, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=finfo.annotation)
        )
        annotations[fname] = finfo.annotation

    def fn(*, ctx: Context, **kwargs: Any) -> dict[str, Any]:
        return run(ctx, spec.name, kwargs)

    fn.__name__ = spec.name
    fn.__doc__ = f"{spec.description} Required scope: {spec.scope.value}."
    fn.__signature__ = inspect.Signature(params, return_annotation=dict[str, Any])  # type: ignore[attr-defined]
    fn.__annotations__ = annotations
    return fn


def build_server(
    *,
    settings: Settings,
    repo_factory: Callable[[], AbstractContextManager[Repository]],
    audit: AuditLog | None = None,
    kill_switch: KillSwitch | None = None,
) -> MCPServer:
    audit = audit or AuditLog()
    kill_switch = kill_switch or KillSwitch()
    server = MCPServer(
        "store-support-tools",
        instructions="Business tools for the store support agents. Every call is scope-checked and tenant-filtered.",
    )

    def run(ctx: Context, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            rc = context_from_headers(dict(ctx.headers or {}), key=settings.token_signing_key)
        except AuthError as e:
            raise ToolError("forbidden", f"auth: {e}") from e
        with repo_factory() as repo:
            return execute_tool(rc, name, args, repo=repo, audit=audit, kill_switch=kill_switch.engaged)

    for spec in TOOLS.values():
        server.tool(name=spec.name, description=f"{spec.description} Required scope: {spec.scope.value}.")(
            _make_tool_fn(spec, run)
        )

    @server.custom_route("/healthz", methods=["GET"])
    async def healthz(_request: Any) -> Any:
        from starlette.responses import JSONResponse

        return JSONResponse({"ok": True, "kill_switch": kill_switch.engaged})

    return server


def main() -> None:
    configure_logging(level=logging.INFO)
    settings = Settings.from_env()
    pool = ConnectionPool(settings.database_url, min_size=1, max_size=5, open=True)

    @contextmanager
    def repo_factory() -> Iterator[Repository]:
        with pool.connection() as conn:
            yield PostgresRepository(conn)

    server = build_server(settings=settings, repo_factory=repo_factory)
    server.run(transport="streamable-http", host="0.0.0.0", port=8765, stateless_http=True, json_response=True)  # noqa: S104


if __name__ == "__main__":
    main()
