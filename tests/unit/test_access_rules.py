"""Access rules are pure functions. These tests are the spec.

Three layers, tested separately:
  1. Entitlement  - does the principal hold the scope the tool requires?
  2. Data authorization - may this principal see THIS row? (tenant + owner)
  3. Row filter   - what WHERE-clause constraints must every query carry?
"""

from __future__ import annotations

import pytest

from store_support.access.identity import AgentIdentity, Principal
from store_support.access.rules import (
    Decision,
    DenyReason,
    Scope,
    authorize_record,
    authorize_tool,
    required_scope,
    row_filter,
)

# ---------- fixtures ----------


@pytest.fixture
def customer() -> Principal:
    return Principal(subject_id="cust-42", tenant_id="store-a", scopes=frozenset({Scope.CUSTOMER_READ_OWN}))


@pytest.fixture
def staff() -> Principal:
    return Principal(subject_id="staff-7", tenant_id="store-a", scopes=frozenset({Scope.STAFF_READ_ALL}))


@pytest.fixture
def manager() -> Principal:
    return Principal(
        subject_id="mgr-1",
        tenant_id="store-a",
        scopes=frozenset({Scope.STAFF_READ_ALL, Scope.MANAGER_REFUND}),
    )


# ---------- 1. entitlement ----------


@pytest.mark.parametrize(
    ("tool", "scope"),
    [
        ("lookup_order", Scope.CUSTOMER_READ_OWN),
        ("check_inventory", Scope.CUSTOMER_READ_OWN),
        ("check_warranty", Scope.CUSTOMER_READ_OWN),
        ("create_ticket", Scope.CUSTOMER_READ_OWN),
        ("schedule_service", Scope.STAFF_READ_ALL),
        ("issue_refund", Scope.MANAGER_REFUND),
    ],
)
def test_every_tool_declares_a_required_scope(tool: str, scope: Scope) -> None:
    assert required_scope(tool) == scope


def test_unknown_tool_is_denied(customer: Principal) -> None:
    d = authorize_tool(customer, "drop_database")
    assert d.allowed is False
    assert d.reason == DenyReason.UNKNOWN_TOOL


def test_customer_can_lookup_order(customer: Principal) -> None:
    assert authorize_tool(customer, "lookup_order").allowed is True


def test_customer_cannot_issue_refund(customer: Principal) -> None:
    d = authorize_tool(customer, "issue_refund")
    assert d.allowed is False
    assert d.reason == DenyReason.MISSING_SCOPE


def test_staff_cannot_issue_refund(staff: Principal) -> None:
    assert authorize_tool(staff, "issue_refund").allowed is False


def test_manager_can_issue_refund(manager: Principal) -> None:
    assert authorize_tool(manager, "issue_refund").allowed is True


def test_staff_read_all_implies_read_own(staff: Principal) -> None:
    """Scope hierarchy: staff:read_all satisfies any tool that needs customer:read_own."""
    assert authorize_tool(staff, "lookup_order").allowed is True


def test_manager_refund_does_not_imply_read_all() -> None:
    """Scopes are explicit; a refund-only token cannot browse other customers."""
    p = Principal(subject_id="mgr-x", tenant_id="store-a", scopes=frozenset({Scope.MANAGER_REFUND}))
    assert authorize_tool(p, "schedule_service").allowed is False


# ---------- 2. data authorization ----------


def test_customer_sees_own_record(customer: Principal) -> None:
    d = authorize_record(customer, owner_id="cust-42", tenant_id="store-a")
    assert d.allowed is True


def test_customer_cannot_see_other_customers_record(customer: Principal) -> None:
    d = authorize_record(customer, owner_id="cust-99", tenant_id="store-a")
    assert d.allowed is False
    assert d.reason == DenyReason.NOT_OWNER


def test_staff_sees_any_record_in_tenant(staff: Principal) -> None:
    assert authorize_record(staff, owner_id="cust-99", tenant_id="store-a").allowed is True


def test_cross_tenant_is_always_denied_even_for_staff(staff: Principal) -> None:
    d = authorize_record(staff, owner_id="cust-99", tenant_id="store-b")
    assert d.allowed is False
    assert d.reason == DenyReason.TENANT_MISMATCH


def test_cross_tenant_denied_before_owner_check(customer: Principal) -> None:
    """Same subject id in another tenant must still be blocked (tenant is checked first)."""
    d = authorize_record(customer, owner_id="cust-42", tenant_id="store-b")
    assert d.reason == DenyReason.TENANT_MISMATCH


# ---------- 3. row filter ----------


def test_customer_row_filter_pins_tenant_and_owner(customer: Principal) -> None:
    f = row_filter(customer)
    assert f.tenant_id == "store-a"
    assert f.owner_id == "cust-42"


def test_staff_row_filter_pins_tenant_only(staff: Principal) -> None:
    f = row_filter(staff)
    assert f.tenant_id == "store-a"
    assert f.owner_id is None


def test_row_filter_always_has_tenant() -> None:
    p = Principal(subject_id="x", tenant_id="store-z", scopes=frozenset())
    assert row_filter(p).tenant_id == "store-z"


# ---------- decisions carry no PII ----------


def test_decision_repr_contains_no_subject_id(customer: Principal) -> None:
    d = authorize_record(customer, owner_id="cust-99", tenant_id="store-a")
    text = repr(d) + str(d)
    assert "cust-42" not in text
    assert "cust-99" not in text


def test_decision_is_immutable(customer: Principal) -> None:
    d = authorize_tool(customer, "lookup_order")
    with pytest.raises((AttributeError, TypeError)):
        d.allowed = False  # type: ignore[misc]


# ---------- identity: agent vs user ----------


def test_agent_identity_is_separate_from_principal(customer: Principal) -> None:
    agent = AgentIdentity(name="knowledge-agent", version="1.0.0")
    assert agent.name != customer.subject_id
    assert not hasattr(agent, "scopes")  # agents never carry user scopes


def test_principal_is_frozen(customer: Principal) -> None:
    with pytest.raises((AttributeError, TypeError)):
        customer.scopes = frozenset()  # type: ignore[misc]


def test_decision_truthiness_matches_allowed() -> None:
    assert bool(Decision(allowed=True, reason=None)) is True
    assert bool(Decision(allowed=False, reason=DenyReason.MISSING_SCOPE)) is False
