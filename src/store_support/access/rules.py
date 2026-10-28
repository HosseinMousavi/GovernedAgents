"""Pure access rules. No I/O, no globals, no logging - just functions of their inputs.

Entitlement  : authorize_tool(principal, tool)        -> may this scope call this tool?
Data auth    : authorize_record(principal, owner, tn) -> may this principal see this row?
Row filter   : row_filter(principal)                  -> constraints every query must carry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from store_support.access.identity import Principal


class Scope(StrEnum):
    CUSTOMER_READ_OWN = "customer:read_own"
    STAFF_READ_ALL = "staff:read_all"
    MANAGER_REFUND = "manager:refund"


class DenyReason(StrEnum):
    UNKNOWN_TOOL = "unknown_tool"
    MISSING_SCOPE = "missing_scope"
    TENANT_MISMATCH = "tenant_mismatch"
    NOT_OWNER = "not_owner"
    KILL_SWITCH = "kill_switch"


@dataclass(frozen=True, slots=True)
class Decision:
    """Deliberately carries only a boolean and a reason *code* - never ids, never PII."""

    allowed: bool
    reason: DenyReason | None

    def __bool__(self) -> bool:
        return self.allowed


ALLOW: Final = Decision(allowed=True, reason=None)


def deny(reason: DenyReason) -> Decision:
    return Decision(allowed=False, reason=reason)


# Which scopes *satisfy* a requirement. staff:read_all is a superset of customer:read_own.
# manager:refund is intentionally NOT a superset of anything: refund rights != browse rights.
_SATISFIES: Final[dict[Scope, frozenset[Scope]]] = {
    Scope.CUSTOMER_READ_OWN: frozenset({Scope.CUSTOMER_READ_OWN, Scope.STAFF_READ_ALL}),
    Scope.STAFF_READ_ALL: frozenset({Scope.STAFF_READ_ALL}),
    Scope.MANAGER_REFUND: frozenset({Scope.MANAGER_REFUND}),
}

TOOL_SCOPES: Final[dict[str, Scope]] = {
    "lookup_order": Scope.CUSTOMER_READ_OWN,
    "check_inventory": Scope.CUSTOMER_READ_OWN,
    "check_warranty": Scope.CUSTOMER_READ_OWN,
    "create_ticket": Scope.CUSTOMER_READ_OWN,
    "schedule_service": Scope.STAFF_READ_ALL,
    "issue_refund": Scope.MANAGER_REFUND,
}


def required_scope(tool: str) -> Scope | None:
    return TOOL_SCOPES.get(tool)


def authorize_tool(principal: Principal, tool: str) -> Decision:
    needed = required_scope(tool)
    if needed is None:
        return deny(DenyReason.UNKNOWN_TOOL)
    if principal.scopes & _SATISFIES[needed]:
        return ALLOW
    return deny(DenyReason.MISSING_SCOPE)


def authorize_record(principal: Principal, *, owner_id: str, tenant_id: str) -> Decision:
    # Tenant isolation is checked FIRST and unconditionally.
    if tenant_id != principal.tenant_id:
        return deny(DenyReason.TENANT_MISMATCH)
    if principal.has(Scope.STAFF_READ_ALL):
        return ALLOW
    if owner_id == principal.subject_id:
        return ALLOW
    return deny(DenyReason.NOT_OWNER)


@dataclass(frozen=True, slots=True)
class RowFilter:
    """Mandatory predicates. tenant_id is never None; owner_id is None only for read_all."""

    tenant_id: str
    owner_id: str | None

    def sql(self, *, tenant_col: str = "tenant_id", owner_col: str = "customer_id") -> tuple[str, list[str]]:
        clauses = [f"{tenant_col} = %s"]
        params = [self.tenant_id]
        if self.owner_id is not None:
            clauses.append(f"{owner_col} = %s")
            params.append(self.owner_id)
        return " AND ".join(clauses), params


def row_filter(principal: Principal) -> RowFilter:
    owner = None if principal.has(Scope.STAFF_READ_ALL) else principal.subject_id
    return RowFilter(tenant_id=principal.tenant_id, owner_id=owner)
