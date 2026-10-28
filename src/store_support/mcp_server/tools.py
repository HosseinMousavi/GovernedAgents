"""Tool registry + single execution path.

Every call goes through `execute_tool`, in this fixed order:
  1. kill switch       (governance)
  2. entitlement       (does the scope allow this tool?)
  3. input validation  (pydantic model == published JSON schema)
  4. data authorization (RowFilter applied in the repository; misses are 'not_found')
  5. business logic
  6. audit             (always, success or failure)

Adding a tool = one pydantic model + one handler + one registry line (see README "add a tool").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from store_support.access import DenyReason, RequestContext, Scope, authorize_tool, required_scope, row_filter
from store_support.mcp_server.audit import AuditLog
from store_support.mcp_server.repo import Order, Repository
from store_support.telemetry.tracing import tool_span


class ToolError(Exception):
    """code: forbidden | invalid_input | not_found | business_rule. Never carries PII."""

    def __init__(self, code: str, message: str, reason: DenyReason | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.reason = reason


# ---------- input models (their JSON schema is what MCP publishes) ----------


class LookupOrderIn(BaseModel):
    order_id: int = Field(ge=1, description="Numeric order id, e.g. 1042")


class CheckInventoryIn(BaseModel):
    sku: str = Field(min_length=1, max_length=64)


class CheckWarrantyIn(BaseModel):
    order_id: int = Field(ge=1)


class CreateTicketIn(BaseModel):
    order_id: int = Field(ge=1)
    category: str = Field(pattern="^(damaged|missing|wrong_item|defective|other)$")
    summary: str = Field(min_length=3, max_length=500)


class ScheduleServiceIn(BaseModel):
    order_id: int = Field(ge=1)
    preferred_date: date


class IssueRefundIn(BaseModel):
    order_id: int = Field(ge=1)
    amount_cents: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=200)


# ---------- handlers (business logic only; auth already done) ----------


def _order_or_404(repo: Repository, ctx: RequestContext, order_id: int) -> Order:
    order = repo.get_order(order_id, row_filter(ctx.principal))
    if order is None:
        raise ToolError("not_found", "order not found")
    return order


def _order_view(o: Order) -> dict[str, Any]:
    return {
        "order_id": o.order_id,
        "sku": o.sku,
        "product_name": o.product_name,
        "status": o.status,
        "total_cents": o.total_cents,
        "refunded_cents": o.refunded_cents,
        "ordered_on": o.ordered_on.isoformat(),
    }


def lookup_order(ctx: RequestContext, repo: Repository, a: LookupOrderIn) -> dict[str, Any]:
    return _order_view(_order_or_404(repo, ctx, a.order_id))


def check_inventory(ctx: RequestContext, repo: Repository, a: CheckInventoryIn) -> dict[str, Any]:
    inv = repo.get_inventory(a.sku, ctx.principal.tenant_id)
    if inv is None:
        raise ToolError("not_found", "sku not found")
    return {
        "sku": inv.sku,
        "product_name": inv.product_name,
        "in_stock": inv.stock > 0,
        "stock": inv.stock,
        "price_cents": inv.price_cents,
    }


def check_warranty(
    ctx: RequestContext, repo: Repository, a: CheckWarrantyIn, *, today: date | None = None
) -> dict[str, Any]:
    o = _order_or_404(repo, ctx, a.order_id)
    today = today or date.today()
    expires = o.ordered_on + timedelta(days=30 * o.warranty_months)
    return {
        "order_id": o.order_id,
        "months": o.warranty_months,
        "expires_on": expires.isoformat(),
        "covered": today <= expires,
    }


def create_ticket(ctx: RequestContext, repo: Repository, a: CreateTicketIn) -> dict[str, Any]:
    o = _order_or_404(repo, ctx, a.order_id)
    tid = repo.create_ticket(order=o, category=a.category, summary=a.summary, created_by=ctx.agent.name)
    return {"ticket_id": tid, "order_id": o.order_id, "status": "open"}


def schedule_service(ctx: RequestContext, repo: Repository, a: ScheduleServiceIn) -> dict[str, Any]:
    o = _order_or_404(repo, ctx, a.order_id)
    if a.preferred_date < date.today() - timedelta(days=1):
        raise ToolError("business_rule", "date is in the past")
    aid = repo.schedule_service(order=o, preferred_date=a.preferred_date, created_by=ctx.agent.name)
    return {"appointment_id": aid, "order_id": o.order_id, "date": a.preferred_date.isoformat()}


def issue_refund(ctx: RequestContext, repo: Repository, a: IssueRefundIn) -> dict[str, Any]:
    o = _order_or_404(repo, ctx, a.order_id)
    if o.refunded_cents > 0:
        raise ToolError("business_rule", "order already refunded")
    if a.amount_cents > o.total_cents:
        raise ToolError("business_rule", "refund exceeds order total")
    rid = repo.issue_refund(order=o, amount_cents=a.amount_cents, reason=a.reason, approved_by=ctx.principal.subject_id)
    return {"refund_id": rid, "order_id": o.order_id, "amount_cents": a.amount_cents, "status": "refunded"}


# ---------- registry ----------


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    model: type[BaseModel]
    handler: Callable[..., dict[str, Any]]

    @property
    def scope(self) -> Scope:
        s = required_scope(self.name)
        assert s is not None, f"{self.name} missing from access.rules.TOOL_SCOPES"
        return s

    def input_schema(self) -> dict[str, Any]:
        return self.model.model_json_schema()


TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in [
        ToolSpec("lookup_order", "Fetch an order the caller is allowed to see.", LookupOrderIn, lookup_order),
        ToolSpec(
            "check_inventory", "Stock and price for a SKU in the caller's store.", CheckInventoryIn, check_inventory
        ),
        ToolSpec("check_warranty", "Warranty window for an order.", CheckWarrantyIn, check_warranty),
        ToolSpec("create_ticket", "Open a support ticket against an order.", CreateTicketIn, create_ticket),
        ToolSpec("schedule_service", "Book a service appointment (staff only).", ScheduleServiceIn, schedule_service),
        ToolSpec("issue_refund", "Refund an order, up to its total (manager only).", IssueRefundIn, issue_refund),
    ]
}


def execute_tool(
    ctx: RequestContext,
    name: str,
    args: dict[str, Any],
    *,
    repo: Repository,
    audit: AuditLog,
    kill_switch: bool = False,
) -> dict[str, Any]:
    with tool_span(name, ctx) as span:
        try:
            if kill_switch:
                raise ToolError("forbidden", "tool execution disabled by kill switch", DenyReason.KILL_SWITCH)
            decision = authorize_tool(ctx.principal, name)
            if not decision:
                raise ToolError("forbidden", "not entitled", decision.reason)
            spec = TOOLS[name]
            try:
                parsed = spec.model.model_validate(args)
            except ValidationError as e:
                raise ToolError("invalid_input", f"{e.error_count()} validation error(s)") from e
            result = spec.handler(ctx, repo, parsed)
        except ToolError as e:
            outcome = "denied" if e.code == "forbidden" else "error"
            span.set_attribute("tool.outcome", outcome)
            span.set_attribute("tool.reason", str(e.reason or e.code))
            audit.record(ctx, tool=name, outcome=outcome, reason=e.reason or e.code)
            raise
        span.set_attribute("tool.outcome", "ok")
        audit.record(ctx, tool=name, outcome="ok")
        return result
