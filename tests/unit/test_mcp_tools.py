"""MCP tool execution: schema validation -> kill switch -> entitlement -> data auth -> business logic.

Uses the in-memory repository; identical contract to the Postgres one.
"""

from __future__ import annotations

import pytest

from store_support.access import AgentIdentity, DenyReason, Principal, RequestContext, Scope
from store_support.mcp_server.audit import AuditLog
from store_support.mcp_server.repo_memory import InMemoryRepository
from store_support.mcp_server.tools import TOOLS, ToolError, execute_tool

AGENT = AgentIdentity(name="knowledge-agent", version="1.0.0")


def ctx(p: Principal) -> RequestContext:
    return RequestContext(principal=p, agent=AGENT, request_id="req-1")


@pytest.fixture
def repo() -> InMemoryRepository:
    r = InMemoryRepository()
    r.add_customer("cust-42", tenant_id="store-a", email="a@example.com")
    r.add_customer("cust-99", tenant_id="store-a", email="b@example.com")
    r.add_customer("cust-b1", tenant_id="store-b", email="c@example.com")
    r.add_product("SKU-1", tenant_id="store-a", name="Steering wheel cover", price_cents=2499, stock=12)
    r.add_order(1042, customer_id="cust-42", tenant_id="store-a", sku="SKU-1", total_cents=2499, warranty_months=12)
    r.add_order(2000, customer_id="cust-99", tenant_id="store-a", sku="SKU-1", total_cents=2499, warranty_months=12)
    r.add_order(3000, customer_id="cust-b1", tenant_id="store-b", sku="SKU-1", total_cents=2499, warranty_months=12)
    return r


@pytest.fixture
def audit() -> AuditLog:
    return AuditLog()


customer = Principal("cust-42", "store-a", frozenset({Scope.CUSTOMER_READ_OWN}))
other_customer = Principal("cust-99", "store-a", frozenset({Scope.CUSTOMER_READ_OWN}))
staff = Principal("staff-7", "store-a", frozenset({Scope.STAFF_READ_ALL}))
manager = Principal("mgr-1", "store-a", frozenset({Scope.STAFF_READ_ALL, Scope.MANAGER_REFUND}))
staff_b = Principal("staff-b", "store-b", frozenset({Scope.STAFF_READ_ALL}))


# ---------- registry ----------


def test_registry_has_six_tools_each_with_schema_and_scope() -> None:
    assert set(TOOLS) == {
        "lookup_order",
        "check_inventory",
        "check_warranty",
        "create_ticket",
        "schedule_service",
        "issue_refund",
    }
    for spec in TOOLS.values():
        schema = spec.input_schema()
        assert schema["type"] == "object"
        assert "properties" in schema
        assert isinstance(spec.scope, Scope)


# ---------- lookup_order ----------


def test_customer_reads_own_order(repo, audit) -> None:
    out = execute_tool(ctx(customer), "lookup_order", {"order_id": 1042}, repo=repo, audit=audit)
    assert out["order_id"] == 1042
    assert out["status"] == "delivered"


def test_customer_cannot_read_others_order_and_cannot_tell_it_exists(repo, audit) -> None:
    with pytest.raises(ToolError) as e:
        execute_tool(ctx(customer), "lookup_order", {"order_id": 2000}, repo=repo, audit=audit)
    assert e.value.code == "not_found"  # no existence leak


def test_staff_reads_any_order_in_tenant(repo, audit) -> None:
    out = execute_tool(ctx(staff), "lookup_order", {"order_id": 2000}, repo=repo, audit=audit)
    assert out["order_id"] == 2000


def test_staff_cannot_read_other_tenants_order(repo, audit) -> None:
    with pytest.raises(ToolError) as e:
        execute_tool(ctx(staff), "lookup_order", {"order_id": 3000}, repo=repo, audit=audit)
    assert e.value.code == "not_found"


def test_order_output_contains_no_email(repo, audit) -> None:
    out = execute_tool(ctx(staff), "lookup_order", {"order_id": 1042}, repo=repo, audit=audit)
    assert "@" not in str(out)


# ---------- validation ----------


def test_schema_validation_rejects_bad_input(repo, audit) -> None:
    with pytest.raises(ToolError) as e:
        execute_tool(ctx(customer), "lookup_order", {"order_id": "not-a-number"}, repo=repo, audit=audit)
    assert e.value.code == "invalid_input"


def test_unknown_tool(repo, audit) -> None:
    with pytest.raises(ToolError) as e:
        execute_tool(ctx(customer), "nuke", {}, repo=repo, audit=audit)
    assert e.value.code == "forbidden"
    assert e.value.reason == DenyReason.UNKNOWN_TOOL


# ---------- issue_refund ----------


def test_customer_cannot_refund(repo, audit) -> None:
    with pytest.raises(ToolError) as e:
        execute_tool(
            ctx(customer),
            "issue_refund",
            {"order_id": 1042, "amount_cents": 2499, "reason": "damaged"},
            repo=repo,
            audit=audit,
        )
    assert e.value.code == "forbidden"
    assert e.value.reason == DenyReason.MISSING_SCOPE


def test_manager_refunds_order_in_tenant(repo, audit) -> None:
    out = execute_tool(
        ctx(manager),
        "issue_refund",
        {"order_id": 1042, "amount_cents": 2499, "reason": "damaged"},
        repo=repo,
        audit=audit,
    )
    assert out["status"] == "refunded"
    assert out["amount_cents"] == 2499


def test_refund_cannot_exceed_order_total(repo, audit) -> None:
    with pytest.raises(ToolError) as e:
        execute_tool(
            ctx(manager),
            "issue_refund",
            {"order_id": 1042, "amount_cents": 99999, "reason": "x"},
            repo=repo,
            audit=audit,
        )
    assert e.value.code == "business_rule"


def test_refund_is_idempotent_per_order(repo, audit) -> None:
    args = {"order_id": 1042, "amount_cents": 2499, "reason": "damaged"}
    execute_tool(ctx(manager), "issue_refund", args, repo=repo, audit=audit)
    with pytest.raises(ToolError) as e:
        execute_tool(ctx(manager), "issue_refund", args, repo=repo, audit=audit)
    assert e.value.code == "business_rule"


def test_manager_cannot_refund_other_tenant(repo, audit) -> None:
    with pytest.raises(ToolError) as e:
        execute_tool(
            ctx(manager), "issue_refund", {"order_id": 3000, "amount_cents": 1, "reason": "x"}, repo=repo, audit=audit
        )
    assert e.value.code == "not_found"


# ---------- create_ticket / schedule_service / inventory / warranty ----------


def test_customer_creates_ticket_for_own_order(repo, audit) -> None:
    out = execute_tool(
        ctx(customer),
        "create_ticket",
        {"order_id": 1042, "category": "damaged", "summary": "arrived damaged"},
        repo=repo,
        audit=audit,
    )
    assert out["ticket_id"].startswith("TCK-")
    assert out["status"] == "open"


def test_customer_cannot_ticket_others_order(repo, audit) -> None:
    with pytest.raises(ToolError):
        execute_tool(
            ctx(customer),
            "create_ticket",
            {"order_id": 2000, "category": "damaged", "summary": "x"},
            repo=repo,
            audit=audit,
        )


def test_customer_cannot_schedule_service(repo, audit) -> None:
    with pytest.raises(ToolError) as e:
        execute_tool(
            ctx(customer),
            "schedule_service",
            {"order_id": 1042, "preferred_date": "2026-11-03"},
            repo=repo,
            audit=audit,
        )
    assert e.value.reason == DenyReason.MISSING_SCOPE


def test_staff_schedules_service(repo, audit) -> None:
    out = execute_tool(
        ctx(staff), "schedule_service", {"order_id": 1042, "preferred_date": "2026-11-03"}, repo=repo, audit=audit
    )
    assert out["appointment_id"].startswith("APT-")


def test_inventory_is_tenant_scoped(repo, audit) -> None:
    out = execute_tool(ctx(customer), "check_inventory", {"sku": "SKU-1"}, repo=repo, audit=audit)
    assert out["in_stock"] is True
    with pytest.raises(ToolError) as e:
        execute_tool(ctx(staff_b), "check_inventory", {"sku": "SKU-1"}, repo=repo, audit=audit)
    assert e.value.code == "not_found"


def test_warranty_for_own_order(repo, audit) -> None:
    out = execute_tool(ctx(customer), "check_warranty", {"order_id": 1042}, repo=repo, audit=audit)
    assert out["covered"] is True
    assert out["months"] == 12


# ---------- kill switch ----------


def test_kill_switch_blocks_every_tool(repo, audit) -> None:
    for name, args in [("lookup_order", {"order_id": 1042}), ("check_inventory", {"sku": "SKU-1"})]:
        with pytest.raises(ToolError) as e:
            execute_tool(ctx(manager), name, args, repo=repo, audit=audit, kill_switch=True)
        assert e.value.reason == DenyReason.KILL_SWITCH


# ---------- audit ----------


def test_audit_logs_agent_and_user_separately_without_pii(repo, audit) -> None:
    execute_tool(ctx(customer), "lookup_order", {"order_id": 1042}, repo=repo, audit=audit)
    entry = audit.entries[-1]
    assert entry["agent"] == "knowledge-agent@1.0.0"
    assert entry["subject"] != "cust-42"  # hashed, not raw
    assert entry["tenant"] == "store-a"
    assert entry["tool"] == "lookup_order"
    assert entry["outcome"] == "ok"
    assert "@example.com" not in str(entry)


def test_audit_records_denials_with_reason_code(repo, audit) -> None:
    with pytest.raises(ToolError):
        execute_tool(
            ctx(customer), "issue_refund", {"order_id": 1042, "amount_cents": 1, "reason": "x"}, repo=repo, audit=audit
        )
    assert audit.entries[-1]["outcome"] == "denied"
    assert audit.entries[-1]["reason"] == "missing_scope"


# ---------- refund atomicity (the repository, not the handler, is the authority) ----------


def test_repository_refuses_second_refund_even_if_handler_check_is_bypassed(repo) -> None:
    from store_support.access import RowFilter
    from store_support.mcp_server.repo import RefundConflictError

    order = repo.get_order(1042, RowFilter("store-a", None))
    repo.issue_refund(order=order, amount_cents=100, reason="a", approved_by="m")
    with pytest.raises(RefundConflictError):
        # stale `order` snapshot (refunded_cents == 0) - simulates a concurrent second approval
        repo.issue_refund(order=order, amount_cents=100, reason="b", approved_by="m")
    assert len(repo.refunds) == 1


def test_handler_maps_refund_conflict_to_business_rule(repo, audit) -> None:
    from store_support.access import RowFilter

    order = repo.get_order(1042, RowFilter("store-a", None))
    repo.issue_refund(order=order, amount_cents=100, reason="a", approved_by="m")
    with pytest.raises(ToolError) as e:
        execute_tool(
            ctx(manager), "issue_refund", {"order_id": 1042, "amount_cents": 1, "reason": "x"}, repo=repo, audit=audit
        )
    assert e.value.code == "business_rule"
