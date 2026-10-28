"""Resolver agent: orchestrates knowledge answer + scoped tool calls into a proposal.

Flow (one trace):
  knowledge.run(question) -> lookup_order -> check_warranty -> decide() -> create_ticket -> Resolution
The resolver NEVER issues refunds; it proposes. n8n routes the proposal to a manager, and the
manager's own token calls issue_refund.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from store_support.access import RequestContext
from store_support.governance.manifest import AgentManifest
from store_support.intake.schema import IssueSpec
from store_support.knowledge.pipeline import Citation, KnowledgeAgent, Outcome
from store_support.mcp_server.audit import AuditLog
from store_support.mcp_server.repo import Repository
from store_support.mcp_server.tools import ToolError, execute_tool
from store_support.resolver.rules import Action, decide
from store_support.telemetry.tracing import request_span, step_span


class KnowledgeLike(Protocol):
    def run(self, question: str) -> Any: ...


class ToolClient(Protocol):
    def call(self, ctx: RequestContext, tool: str, args: dict[str, Any]) -> dict[str, Any]: ...


class LocalToolClient:
    """In-process: same execute_tool path as the server. Used by tests and offline evals."""

    def __init__(self, repo: Repository, audit: AuditLog, kill_switch: bool = False) -> None:
        self._repo, self._audit, self._ks = repo, audit, kill_switch

    def call(self, ctx: RequestContext, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        return execute_tool(ctx, tool, args, repo=self._repo, audit=self._audit, kill_switch=self._ks)


@dataclass(slots=True)
class Resolution:
    action: Action
    amount_cents: int = 0
    requires_manager: bool = False
    rationale: str = ""
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    order: dict[str, Any] | None = None
    ticket_id: str | None = None
    reason: str | None = None  # escalation reason code

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "amount_cents": self.amount_cents,
            "requires_manager": self.requires_manager,
            "rationale": self.rationale,
            "answer": self.answer,
            "citations": [f"{c.source}#{c.section}" for c in self.citations],
            "order": self.order,
            "ticket_id": self.ticket_id,
            "reason": self.reason,
        }


@dataclass(slots=True)
class ResolverAgent:
    manifest: AgentManifest
    knowledge: KnowledgeLike | KnowledgeAgent
    tools: ToolClient
    today: date | None = None

    def call_tool(self, ctx: RequestContext, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if not self.manifest.allows(tool):
            raise PermissionError(f"tool '{tool}' not in manifest for {self.manifest.name}")
        return self.tools.call(ctx, tool, args)

    def resolve(self, ctx: RequestContext, spec: IssueSpec) -> Resolution:
        with request_span(ctx):
            return self._resolve(ctx, spec)

    def _resolve(self, ctx: RequestContext, spec: IssueSpec) -> Resolution:
        today = self.today or date.today()
        product = f" ({spec.product_hint})" if spec.product_hint else ""
        question = f"{spec.category} item{product}, customer wants {spec.desired_outcome}: {spec.description}"

        with step_span("knowledge", agent=self.manifest.name):
            k = self.knowledge.run(question)
        answer = k.answer if k.outcome == Outcome.ANSWERED else ""
        citations = list(k.citations) if k.outcome == Outcome.ANSWERED else []

        try:
            order = self.call_tool(ctx, "lookup_order", {"order_id": spec.order_id})
            warranty = self.call_tool(ctx, "check_warranty", {"order_id": spec.order_id})
        except ToolError as e:
            return Resolution(
                Action.ESCALATE,
                answer=answer,
                citations=citations,
                reason=e.code,
                rationale="could not verify the order for this customer",
            )

        in_stock: bool | None = None
        if spec.desired_outcome == "replacement":
            try:
                in_stock = bool(self.call_tool(ctx, "check_inventory", {"sku": order["sku"]})["in_stock"])
            except ToolError:
                in_stock = None

        with step_span("decide", agent=self.manifest.name):
            d = decide(spec, order, today=today, in_stock=in_stock)

        ticket_id = None
        if d.action != Action.ANSWER_ONLY:
            t = self.call_tool(
                ctx,
                "create_ticket",
                {
                    "order_id": spec.order_id,
                    "category": spec.category,
                    "summary": f"{d.action.value}: {d.rationale}. warranty_covered={warranty['covered']}",
                },
            )
            ticket_id = t["ticket_id"]

        return Resolution(
            d.action,
            d.amount_cents,
            d.requires_manager,
            d.rationale,
            answer,
            citations,
            order={k2: v for k2, v in order.items() if k2 != "customer_id"},
            ticket_id=ticket_id,
        )
