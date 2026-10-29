"""End-to-end over real streamable-HTTP: token in header -> scope check -> tool result.
Uses the in-memory repository; no Postgres needed."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from store_support.access import Scope, issue_token
from store_support.config import Settings
from store_support.governance.kill_switch import KillSwitch
from store_support.mcp_server.audit import AuditLog
from store_support.mcp_server.repo_memory import InMemoryRepository
from store_support.mcp_server.server import build_server

KEY = "itest-key"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def server_url(tmp_path_factory) -> Iterator[tuple[str, AuditLog, KillSwitch, object]]:
    repo = InMemoryRepository()
    repo.add_customer("cust-42", tenant_id="store-a", email="x@y.z")
    repo.add_product("SKU-1", tenant_id="store-a", name="Steering wheel cover", price_cents=2499, stock=3)
    repo.add_order(1042, customer_id="cust-42", tenant_id="store-a", sku="SKU-1", total_cents=2499, warranty_months=12)
    audit = AuditLog()
    flag = tmp_path_factory.mktemp("ks") / "KILL"
    ks = KillSwitch(env={}, flag_file=flag)

    @contextmanager
    def repo_factory() -> Iterator[InMemoryRepository]:
        yield repo

    settings = replace(Settings.from_env(), token_signing_key=KEY)
    app = build_server(settings=settings, repo_factory=repo_factory, audit=audit, kill_switch=ks).streamable_http_app(
        stateless_http=True
    )
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz").status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/mcp", audit, ks, flag
    server.should_exit = True


def _token(*scopes: Scope, sub: str = "cust-42") -> str:
    return issue_token(subject_id=sub, tenant_id="store-a", scopes=list(scopes), key=KEY)


async def _call(url: str, token: str, tool: str, args: dict) -> object:
    headers = {"Authorization": f"Bearer {token}", "X-Agent-Name": "itest-agent", "X-Agent-Version": "0.1"}
    async with httpx.AsyncClient(headers=headers, timeout=10) as http:
        async with streamable_http_client(url, http_client=http) as (r, w, *_):
            async with ClientSession(r, w) as s:
                await s.initialize()
                return await s.call_tool(tool, args)


async def test_lists_six_tools_with_schemas(server_url) -> None:
    url, *_ = server_url
    async with httpx.AsyncClient(timeout=10) as http:
        async with streamable_http_client(url, http_client=http) as (r, w, *_):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tools = (await s.list_tools()).tools
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {
        "lookup_order",
        "check_inventory",
        "check_warranty",
        "create_ticket",
        "schedule_service",
        "issue_refund",
    }
    assert by_name["issue_refund"].input_schema["required"] == ["order_id", "amount_cents", "reason"]
    assert "manager:refund" in (by_name["issue_refund"].description or "")


async def test_customer_token_reads_own_order(server_url) -> None:
    url, audit, *_ = server_url
    res = await _call(url, _token(Scope.CUSTOMER_READ_OWN), "lookup_order", {"order_id": 1042})
    assert res.is_error is False
    assert res.structured_content["order_id"] == 1042
    assert audit.entries[-1]["agent"] == "itest-agent@0.1"


async def test_customer_token_denied_refund(server_url) -> None:
    url, audit, *_ = server_url
    res = await _call(
        url, _token(Scope.CUSTOMER_READ_OWN), "issue_refund", {"order_id": 1042, "amount_cents": 1, "reason": "x"}
    )
    assert res.is_error is True
    assert "forbidden: missing_scope" in res.content[0].text
    assert audit.entries[-1]["reason"] == "missing_scope"


async def test_bad_token_rejected(server_url) -> None:
    url, *_ = server_url
    res = await _call(url, "garbage.token", "lookup_order", {"order_id": 1042})
    assert res.is_error is True


async def test_kill_switch_flips_at_runtime(server_url) -> None:
    url, audit, _ks, flag = server_url
    flag.write_text("1")
    try:
        res = await _call(url, _token(Scope.CUSTOMER_READ_OWN), "check_inventory", {"sku": "SKU-1"})
        assert res.is_error is True
        assert audit.entries[-1]["reason"] == "kill_switch"
    finally:
        flag.unlink()
    res = await _call(url, _token(Scope.CUSTOMER_READ_OWN), "check_inventory", {"sku": "SKU-1"})
    assert res.is_error is False
