"""The demo flow with zero infrastructure: in-memory tools, hashing embedder, extractive LLM.

    python scripts/demo_offline.py

Same code paths as production (execute_tool, LangGraph knowledge agent, resolver rules,
audit log, OTel spans); only the adapters are swapped. Good for a laptop with no Docker.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from store_support.access import AgentIdentity, Principal, RequestContext, Scope
from store_support.evals.runner import ExtractiveLLM, _offline_store, _seed_repo, run_offline
from store_support.governance.logging_setup import configure_logging
from store_support.governance.manifest import load_manifest
from store_support.intake.schema import IssueSpec, missing_fields
from store_support.knowledge.langgraph_agent import build_graph, run_graph
from store_support.mcp_server.audit import AuditLog
from store_support.mcp_server.tools import ToolError, execute_tool
from store_support.resolver.agent import LocalToolClient, ResolverAgent
from store_support.telemetry import tracing

ROOT = Path(__file__).resolve().parents[1]


def banner(n: int, text: str) -> None:
    print(f"\n\033[1;36m## {n}. {text}\033[0m")


def main() -> None:
    configure_logging(level=logging.WARNING)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing._tracer = provider.get_tracer("demo")  # noqa: SLF001

    repo, audit = _seed_repo(), AuditLog()
    graph = build_graph(store=_offline_store(), llm=ExtractiveLLM(), tenant_id="store-a", threshold=0.3)
    manifest = load_manifest(ROOT / "agents" / "resolver-agent" / "manifest.yaml")

    class K:
        def run(self, q: str):  # noqa: ANN202
            return run_graph(graph, q)

    agent = ResolverAgent(
        manifest=manifest, knowledge=K(), tools=LocalToolClient(repo, audit), today=date(2026, 10, 27)
    )
    customer = Principal("cust-0001", "store-a", frozenset({Scope.CUSTOMER_READ_OWN}))
    manager = Principal("mgr-0001", "store-a", frozenset({Scope.STAFF_READ_ALL, Scope.MANAGER_REFUND}))
    ctx = RequestContext(customer, AgentIdentity(manifest.name, manifest.version), "demo-req-1")

    banner(1, 'Customer: "my steering wheel cover arrived damaged, order 1042"  ->  intake spec')
    partial = {"order_id": 1042, "description": "my steering wheel cover arrived damaged"}
    print("  n8n validate ->", {"ok": False, "missing": missing_fields(partial)}, " (re-asks)")
    spec = IssueSpec(
        order_id=1042,
        category="damaged",
        description="my steering wheel cover arrived damaged",
        desired_outcome="refund",
        product_hint="steering wheel cover",
    )
    print("  spec:", spec.model_dump())

    banner(2, "Resolver under the CUSTOMER token: knowledge agent cites policy, lookup_order, proposal")
    res = agent.resolve(ctx, spec)
    print("  answer   :", res.answer[:160].replace("\n", " "), "...")
    print("  citations:", [f"{c.source}#{c.section}" for c in res.citations])
    print("  order    :", {k: res.order[k] for k in ("order_id", "product_name", "status", "total_cents")})
    amount = f"${res.amount_cents / 100:.2f}"
    print(f"  proposal : {res.action.value}  amount={amount}  requires_manager={res.requires_manager}")
    print("  ticket   :", res.ticket_id)

    banner(3, "Customer tries issue_refund directly -> denied")
    try:
        execute_tool(
            ctx,
            "issue_refund",
            {"order_id": 1042, "amount_cents": res.amount_cents, "reason": "x"},
            repo=repo,
            audit=audit,
        )
    except ToolError as e:
        print(f"  {e.code}: {e.reason}")

    banner(4, "Manager approves in n8n -> issue_refund under the MANAGER token")
    mctx = RequestContext(manager, AgentIdentity("n8n-approval-flow", "1.0.0"), "demo-req-1")
    with tracing.request_span(mctx):
        out = execute_tool(
            mctx,
            "issue_refund",
            {"order_id": 1042, "amount_cents": res.amount_cents, "reason": "manager approved: damaged on arrival"},
            repo=repo,
            audit=audit,
        )
    print("  ", out)

    banner(5, "Access log (agent and user logged separately; subject pseudonymized; no PII)")
    for e in audit.entries:
        reason = str(e["reason"] or "")
        who = f"agent={e['agent']:<24} user={e['subject']} tenant={e['tenant']}"
        print(f"  {e['tool']:<16} {e['outcome']:<7} {reason:<14} {who}")

    banner(6, "Trace (one trace per request, span per step and tool call)")
    spans = exporter.get_finished_spans()
    for s in spans:
        tid = format(s.context.trace_id, "032x")[:8]
        print(f"  {s.name:<22} trace={tid}  agent={s.attributes.get('agent.name', '')}")

    banner(7, "Eval scorecard")
    card = run_offline(out_dir=ROOT / "eval_results")
    for k, v in card["metrics"].items():
        print(f"  {k:<24}{v:.3f}")

    banner(8, "Flip the KILL_SWITCH -> every tool call refused, even for the manager")
    try:
        execute_tool(mctx, "lookup_order", {"order_id": 1042}, repo=repo, audit=audit, kill_switch=True)
    except ToolError as e:
        print(f"  {e.code}: {e.reason}")
    print("\n  access log tail:", json.dumps({k: audit.entries[-1][k] for k in ("tool", "outcome", "reason", "agent")}))


if __name__ == "__main__":
    main()
