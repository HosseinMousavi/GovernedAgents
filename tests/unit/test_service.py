"""Agents HTTP service: the intake endpoints n8n uses, auth on /resolve, dev-token gating."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from store_support.config import Settings
from store_support.resolver.service import build_app


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("DEV_TOKENS", "false")
    return TestClient(build_app(Settings.from_env()))


def test_intake_schema_served(client: TestClient) -> None:
    r = client.get("/intake/schema")
    assert r.status_code == 200 and "order_id" in r.json()["properties"]


def test_intake_validate_reports_missing(client: TestClient) -> None:
    r = client.post("/intake/validate", json={"spec": {"order_id": 1042}})
    assert r.json() == {"ok": False, "missing": ["category", "description", "desired_outcome"], "errors": []}


def test_intake_validate_ok(client: TestClient) -> None:
    r = client.post(
        "/intake/validate",
        json={
            "order_id": 1042,
            "category": "damaged",
            "description": "cover arrived torn",
            "desired_outcome": "refund",
        },
    )
    assert r.json()["ok"] is True


def test_resolve_requires_token(client: TestClient) -> None:
    assert client.post("/resolve", json={}).status_code == 401


def test_dev_tokens_disabled_by_default(client: TestClient) -> None:
    assert client.post("/tokens/dev", json={"subject_id": "x", "tenant_id": "t", "scopes": []}).status_code == 404


def test_manifests_listed(client: TestClient) -> None:
    assert set(client.get("/manifests").json()) == {"intake-agent", "knowledge-agent", "resolver-agent"}
