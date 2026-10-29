"""Resolver: IssueSpec + policy answer + order facts -> proposed action. Pure decision rules,
manifest-enforced tool allow-list, and the demo flow end to end over the in-memory tools."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from store_support.access import AgentIdentity, Principal, RequestContext, Scope
from store_support.governance.manifest import load_manifest
from store_support.intake.schema import IssueSpec
from store_support.knowledge.chunking import chunk_markdown
from store_support.knowledge.embeddings import HashEmbedder
from store_support.knowledge.llm import FakeLLM
from store_support.knowledge.pipeline import KnowledgeAgent
from store_support.knowledge.store import InMemoryVectorStore
from store_support.mcp_server.audit import AuditLog
from store_support.mcp_server.repo_memory import InMemoryRepository
from store_support.resolver.agent import LocalToolClient, ResolverAgent
from store_support.resolver.rules import Action, decide

ROOT = Path(__file__).resolve().parents[2]
TODAY = date(2026, 10, 27)


def spec(**kw) -> IssueSpec:
    base = {
        "order_id": 1042,
        "category": "damaged",
        "description": "steering wheel cover arrived damaged",
        "desired_outcome": "refund",
        "product_hint": "steering wheel cover",
    }
    return IssueSpec(**{**base, **kw})


def order(**kw) -> dict:
    base = {
        "order_id": 1042,
        "status": "delivered",
        "total_cents": 2499,
        "refunded_cents": 0,
        "ordered_on": "2026-10-21",
    }
    return {**base, **kw}


# ---------- pure rules ----------


def test_damaged_delivered_within_7_days_refund_requested_proposes_full_refund() -> None:
    d = decide(spec(), order(), today=TODAY)
    assert d.action == Action.PROPOSE_REFUND
    assert d.amount_cents == 2499
    assert d.requires_manager is True


def test_damaged_but_reported_late_goes_to_ticket_only() -> None:
    d = decide(spec(), order(ordered_on="2026-01-01"), today=TODAY)
    assert d.action == Action.TICKET_ONLY
    assert "7 days" in d.rationale


def test_replacement_requested_in_stock_proposes_replacement() -> None:
    d = decide(spec(desired_outcome="replacement"), order(), today=TODAY, in_stock=True)
    assert d.action == Action.PROPOSE_REPLACEMENT
    assert d.requires_manager is False


def test_replacement_requested_out_of_stock_falls_back_to_refund() -> None:
    d = decide(spec(desired_outcome="replacement"), order(), today=TODAY, in_stock=False)
    assert d.action == Action.PROPOSE_REFUND


def test_already_refunded_order_is_ticket_only() -> None:
    d = decide(spec(), order(status="refunded", refunded_cents=2499), today=TODAY)
    assert d.action == Action.TICKET_ONLY


def test_not_delivered_yet_is_ticket_only() -> None:
    d = decide(spec(category="missing"), order(status="shipped"), today=TODAY)
    assert d.action == Action.TICKET_ONLY


def test_information_only_never_proposes_money() -> None:
    d = decide(spec(desired_outcome="information"), order(), today=TODAY)
    assert d.action == Action.ANSWER_ONLY
    assert d.amount_cents == 0


# ---------- end-to-end over in-memory tools ----------


@pytest.fixture
def world():
    repo = InMemoryRepository()
    repo.add_customer("cust-0001", tenant_id="store-a", email="a@b.c")
    repo.add_product("SKU-001", tenant_id="store-a", name="Steering Wheel Cover - Leather", price_cents=2499, stock=5)
    repo.add_order(
        1042,
        customer_id="cust-0001",
        tenant_id="store-a",
        sku="SKU-001",
        total_cents=2499,
        warranty_months=12,
        ordered_on=date(2026, 10, 21),
    )
    store = InMemoryVectorStore(HashEmbedder(dim=128))
    for f in (ROOT / "data" / "policies").glob("*.md"):
        store.add("store-a", chunk_markdown(f.read_text(), source=f.name))
    llm = FakeLLM(
        grades=[0.9],
        answer="Damaged on arrival: report within 7 days for a full refund or replacement. "
        "[manual-steering-wheel-cover.md#Common issues]",
    )
    knowledge = KnowledgeAgent(store=store, llm=llm, tenant_id="store-a")
    audit = AuditLog()
    manifest = load_manifest(ROOT / "agents" / "resolver-agent" / "manifest.yaml")
    principal = Principal("cust-0001", "store-a", frozenset({Scope.CUSTOMER_READ_OWN}))
    ctx = RequestContext(principal=principal, agent=AgentIdentity(manifest.name, manifest.version), request_id="r1")
    client = LocalToolClient(repo=repo, audit=audit)
    return ResolverAgent(manifest=manifest, knowledge=knowledge, tools=client, today=TODAY), ctx, audit, repo


def test_demo_flow_proposes_refund_with_citation_and_ticket(world) -> None:
    agent, ctx, audit, repo = world
    res = agent.resolve(ctx, spec())
    assert res.action == Action.PROPOSE_REFUND
    assert res.amount_cents == 2499
    assert res.citations[0].source == "manual-steering-wheel-cover.md"
    assert res.ticket_id and repo.tickets[0]["category"] == "damaged"
    tools_called = [e["tool"] for e in audit.entries]
    assert tools_called == ["lookup_order", "check_warranty", "create_ticket"]
    assert all(e["agent"] == "resolver-agent@1.0.0" for e in audit.entries)


def test_resolver_never_calls_issue_refund_even_if_asked(world) -> None:
    agent, ctx, *_ = world
    with pytest.raises(PermissionError, match="manifest"):
        agent.call_tool(ctx, "issue_refund", {"order_id": 1042, "amount_cents": 1, "reason": "x"})


def test_resolver_on_foreign_order_escalates_without_leaking(world) -> None:
    agent, ctx, audit, repo = world
    repo.add_customer("cust-0002", tenant_id="store-a", email="z@z.z")
    repo.add_order(
        2000, customer_id="cust-0002", tenant_id="store-a", sku="SKU-001", total_cents=2499, warranty_months=12
    )
    res = agent.resolve(ctx, spec(order_id=2000))
    assert res.action == Action.ESCALATE
    assert res.reason == "not_found"
    assert "cust-0002" not in str(res)


# ---------- cost surfaces + budget exhaustion ----------


def test_resolution_carries_knowledge_cost(world) -> None:
    agent, ctx, *_ = world
    res = agent.resolve(ctx, spec())
    assert res.cost_usd == 0.0 and res.models == [] and res.to_dict()["cost_usd"] == 0.0


def test_hard_budget_escalates_to_human_instead_of_failing(world) -> None:
    from store_support.knowledge.llm import BudgetExhaustedError

    agent, ctx, audit, repo = world

    class Broke:
        def run(self, q):  # noqa: ANN001, ANN202
            raise BudgetExhaustedError("tenant daily budget exhausted")

    agent.knowledge = Broke()
    res = agent.resolve(ctx, spec())
    assert res.action == Action.ESCALATE
    assert res.reason == "budget_exhausted"
    assert [e["tool"] for e in audit.entries] == []  # no tool calls once we know we must hand off
