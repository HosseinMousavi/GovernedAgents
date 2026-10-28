"""ToolClient over the real MCP server (streamable HTTP, stateless).

The user's scoped token and the agent's identity travel as HTTP headers; the server
re-derives the Principal from the token, so the agent cannot upgrade its own scopes.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from store_support.access import RequestContext
from store_support.mcp_server.tools import ToolError


class McpToolClient:
    def __init__(self, url: str, *, token_for: Any) -> None:
        """token_for(ctx) -> str: resolves the bearer token for the principal in ctx."""
        self._url = url
        self._token_for = token_for

    def call(self, ctx: RequestContext, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self._call(ctx, tool, args))

    async def _call(self, ctx: RequestContext, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._token_for(ctx)}",
            "X-Agent-Name": ctx.agent.name,
            "X-Agent-Version": ctx.agent.version,
            "X-Request-Id": ctx.request_id,
        }
        async with httpx.AsyncClient(headers=headers, timeout=30) as http:
            async with streamable_http_client(self._url, http_client=http) as (r, w, *_):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    res = await s.call_tool(tool, args)
        if res.is_error:
            text = "".join(getattr(c, "text", "") for c in res.content)
            code = "forbidden" if "forbidden" in text or "not entitled" in text or "kill switch" in text else (
                "not_found" if "not found" in text else "error")
            raise ToolError(code, text[:200])
        if res.structured_content:
            return dict(res.structured_content)
        return json.loads("".join(getattr(c, "text", "") for c in res.content))  # type: ignore[no-any-return]
