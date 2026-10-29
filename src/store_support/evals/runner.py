"""`make eval` -> scorecard.

Two modes, same datasets and metrics:
  offline (default)  HashEmbedder + ExtractiveLLM + in-memory store/tools. Zero network. Runs in CI.
  online  (--online) Real embeddings/LLM/Postgres/MCP via Settings. Run before a release.

Metrics: retrieval_hit_rate, faithfulness, tool_choice_accuracy, scope_enforcement.
Writes eval_results/latest.json (and baseline.json if missing) for the weekly drift check.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import typer

from store_support.access import AgentIdentity, Principal, RequestContext, Scope
from store_support.evals.metrics import faithfulness, retrieval_hit, scope_enforced, tool_choice
from store_support.governance.cost import Usage
from store_support.governance.manifest import load_manifest
from store_support.intake.schema import IssueSpec
from store_support.knowledge.chunking import chunk_markdown
from store_support.knowledge.embeddings import HashEmbedder
from store_support.knowledge.langgraph_agent import build_graph, run_graph
from store_support.knowledge.store import Hit, InMemoryVectorStore
from store_support.mcp_server.audit import AuditLog
from store_support.mcp_server.repo_memory import InMemoryRepository
from store_support.mcp_server.tools import ToolError, execute_tool
from store_support.resolver.agent import LocalToolClient, ResolverAgent

ROOT = Path(__file__).resolve().parents[3]
DATASETS = Path(__file__).resolve().parent / "datasets"
TODAY = date(2026, 10, 27)


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    question: str
    expected_source: str
    keywords: list[str]


@dataclass(frozen=True, slots=True)
class Scenario:
    id: str
    name: str
    expected_action: str
    expected_tools: list[str]
    expected_denied: bool
    principal: dict[str, Any] | None = None
    spec: dict[str, Any] | None = None
    tool_call: dict[str, Any] | None = None


def load_questions(path: Path) -> list[Question]:
    return [Question(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def load_scenarios(path: Path) -> list[Scenario]:
    return [Scenario(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


class ExtractiveLLM:
    """Offline stand-in: grades by keyword overlap, answers by quoting the best chunk with its tag.
    Deterministic, so offline eval numbers measure retrieval + pipeline, not an API."""

    _TOKEN = re.compile(r"[a-z0-9.$:-]+")

    def _overlap(self, q: str, text: str) -> float:
        qs = {t for t in self._TOKEN.findall(q.lower()) if len(t) > 2}
        ts = set(self._TOKEN.findall(text.lower()))
        return len(qs & ts) / max(1, len(qs))

    def grade(self, question: str, hits: Sequence[Hit]) -> float:
        return min(1.0, max((self._overlap(question, h.content) for h in hits), default=0.0) * 2)

    def rewrite(self, question: str, previous: Sequence[str]) -> str:
        return question + " policy refund return warranty shipping"

    def filter(self, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Keep the top-3 by overlap (offline stand-in for the cheap filter model)."""
        ranked = sorted(hits, key=lambda h: self._overlap(question, h.content), reverse=True)
        return ranked[:3]

    def answer(self, question: str, hits: Sequence[Hit], *, attempt: int = 0, last_score: float | None = None) -> str:
        best = max(hits, key=lambda h: self._overlap(question, h.content))
        body = best.content.split("\n", 1)[-1][:400]
        return f"{body} {best.citation}"

    def drain_usage(self) -> tuple[Usage, float, list[str]]:
        return Usage(), 0.0, []


def _principal(p: dict[str, Any]) -> Principal:
    return Principal(p["subject"], p["tenant"], frozenset(Scope(s) for s in p["scopes"]))


def _seed_repo() -> InMemoryRepository:
    r = InMemoryRepository()
    r.add_customer("cust-0001", tenant_id="store-a", email="a@x.y")
    r.add_customer("cust-0002", tenant_id="store-a", email="b@x.y")
    r.add_customer("cust-b001", tenant_id="store-b", email="c@x.y")
    r.add_product("SKU-001", tenant_id="store-a", name="Steering Wheel Cover - Leather", price_cents=2499, stock=5)
    r.add_product("SKU-003", tenant_id="store-a", name="All-Weather Floor Mats (set)", price_cents=7999, stock=0)
    r.add_product("SKU-001", tenant_id="store-b", name="Steering Wheel Cover - Leather", price_cents=2499, stock=5)
    recent, old = date(2026, 10, 21), date(2026, 6, 1)
    r.add_order(
        1042,
        customer_id="cust-0001",
        tenant_id="store-a",
        sku="SKU-001",
        total_cents=2499,
        warranty_months=12,
        ordered_on=recent,
    )
    r.add_order(
        1050,
        customer_id="cust-0001",
        tenant_id="store-a",
        sku="SKU-001",
        total_cents=2499,
        warranty_months=12,
        ordered_on=old,
    )
    r.add_order(
        1051,
        customer_id="cust-0001",
        tenant_id="store-a",
        sku="SKU-001",
        total_cents=2499,
        warranty_months=12,
        ordered_on=recent,
        status="shipped",
    )
    r.add_order(
        1052,
        customer_id="cust-0001",
        tenant_id="store-a",
        sku="SKU-001",
        total_cents=2499,
        warranty_months=12,
        ordered_on=recent,
        status="refunded",
    )
    r.add_order(
        1053,
        customer_id="cust-0001",
        tenant_id="store-a",
        sku="SKU-003",
        total_cents=7999,
        warranty_months=24,
        ordered_on=recent,
    )
    r.add_order(
        2000,
        customer_id="cust-0002",
        tenant_id="store-a",
        sku="SKU-001",
        total_cents=2499,
        warranty_months=12,
        ordered_on=recent,
    )
    r.add_order(
        3000,
        customer_id="cust-b001",
        tenant_id="store-b",
        sku="SKU-001",
        total_cents=2499,
        warranty_months=12,
        ordered_on=recent,
    )
    # s09 expects "already refunded": mark refunded_cents
    from dataclasses import replace

    r._orders[1052] = replace(r._orders[1052], refunded_cents=2499)  # noqa: SLF001 - eval fixture
    return r


def _offline_store() -> InMemoryVectorStore:
    s = InMemoryVectorStore(HashEmbedder(dim=512))
    for f in sorted((ROOT / "data" / "policies").glob("*.md")):
        s.add("store-a", chunk_markdown(f.read_text(), source=f.name))
    return s


def run_offline(*, out_dir: Path | None = None, threshold: float = 0.3) -> dict[str, Any]:
    store, llm = _offline_store(), ExtractiveLLM()
    graph = build_graph(store=store, llm=llm, tenant_id="store-a", k=5, threshold=threshold)

    q_rows = []
    for q in load_questions(DATASETS / "questions.jsonl"):
        res = run_graph(graph, q.question)
        hit = retrieval_hit(res.hits, q.expected_source)
        faith = faithfulness(res.answer, res.citations, res.hits, q.keywords)
        q_rows.append(
            {
                "id": q.id,
                "hit": hit,
                "faithfulness": faith,
                "outcome": res.outcome.value,
                "retries": res.retries,
                "cost_usd": res.cost_usd,
                "tokens": res.usage.total_tokens,
                "context_chunks": len(res.hits),
                "context_used": len(res.hits_used),
            }
        )

    manifest = load_manifest(ROOT / "agents" / "resolver-agent" / "manifest.yaml")
    s_rows = []
    for sc in load_scenarios(DATASETS / "scenarios.jsonl"):
        repo, audit = _seed_repo(), AuditLog()
        if sc.tool_call:
            tc = sc.tool_call
            ctx = RequestContext(_principal(tc["principal"]), AgentIdentity("eval-direct", "1"), sc.id)
            try:
                execute_tool(
                    ctx, tc["tool"], tc["args"], repo=repo, audit=audit, kill_switch=bool(tc.get("kill_switch"))
                )
                observed_action, denied = "ok", False
            except ToolError as e:
                observed_action = "denied" if e.code == "forbidden" else e.code
                denied = e.code in {"forbidden", "not_found"}
        else:
            assert sc.principal and sc.spec
            ctx = RequestContext(_principal(sc.principal), AgentIdentity(manifest.name, manifest.version), sc.id)
            agent = ResolverAgent(
                manifest=manifest, knowledge=_Knowledge(graph), tools=LocalToolClient(repo, audit), today=TODAY
            )
            res = agent.resolve(ctx, IssueSpec(**sc.spec))
            observed_action = res.action.value
            denied = res.reason in {"not_found", "forbidden"}
        tools_seen = [e["tool"] for e in audit.entries]
        s_rows.append(
            {
                "id": sc.id,
                "name": sc.name,
                "expected": sc.expected_action,
                "observed": observed_action,
                "action_ok": observed_action == sc.expected_action,
                "tools_ok": tool_choice(tools_seen, sc.expected_tools),
                "scope_ok": scope_enforced(expected_denied=sc.expected_denied, observed_denied=denied),
            }
        )

    card = {
        "mode": "offline",
        "ts": time.time(),
        "n_questions": len(q_rows),
        "n_scenarios": len(s_rows),
        "metrics": {
            "retrieval_hit_rate": _mean([r["hit"] for r in q_rows]),
            "faithfulness": _mean([r["faithfulness"] for r in q_rows]),
            "tool_choice_accuracy": _mean([r["tools_ok"] and r["action_ok"] for r in s_rows]),
            "scope_enforcement": _mean([r["scope_ok"] for r in s_rows]),
        },
        "cost": {
            "avg_cost_usd": round(sum(r["cost_usd"] for r in q_rows) / max(1, len(q_rows)), 6),
            "avg_tokens": round(sum(r["tokens"] for r in q_rows) / max(1, len(q_rows)), 1),
            "avg_context_compaction": round(
                _mean([1 - r["context_used"] / r["context_chunks"] for r in q_rows if r["context_chunks"]]), 3
            ),
        },
        "questions": q_rows,
        "scenarios": s_rows,
    }
    out_dir = out_dir or ROOT / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(card, indent=2))
    if not (out_dir / "baseline.json").exists():
        (out_dir / "baseline.json").write_text(json.dumps(card, indent=2))
    return card


class _Knowledge:
    def __init__(self, graph: Any) -> None:
        self._g = graph

    def run(self, question: str) -> Any:
        return run_graph(self._g, question)


def _mean(xs: Sequence[float | bool]) -> float:
    return round(sum(float(x) for x in xs) / len(xs), 3) if xs else 0.0


def print_scorecard(card: dict[str, Any]) -> None:
    m = card["metrics"]
    print(f"\n=== Store Support Agent - eval scorecard ({card['mode']}) ===")
    print(f"{'metric':<24}{'score':>8}   target")
    for k, target in [
        ("retrieval_hit_rate", 0.8),
        ("faithfulness", 0.8),
        ("tool_choice_accuracy", 1.0),
        ("scope_enforcement", 1.0),
    ]:
        flag = "ok " if m[k] >= target else "LOW"
        print(f"{k:<24}{m[k]:>8.3f}   >= {target:.2f}  {flag}")
    c = card.get("cost", {})
    if c:
        print(
            f"\navg cost/question ${c['avg_cost_usd']:.4f}   avg tokens {c['avg_tokens']:.0f}   "
            f"context compaction {c['avg_context_compaction']:.0%}"
        )
    misses = [r["id"] for r in card["questions"] if not r["hit"]]
    bad = [r["id"] for r in card["scenarios"] if not (r["tools_ok"] and r["action_ok"] and r["scope_ok"])]
    print(f"\nquestions: {card['n_questions']}  retrieval misses: {misses or 'none'}")
    print(f"scenarios: {card['n_scenarios']}  failures: {bad or 'none'}\n")


app = typer.Typer(add_completion=False)


@app.command()
def main(online: bool = False) -> None:
    import logging

    from store_support.governance.logging_setup import configure_logging

    configure_logging(level=logging.WARNING)  # keep the scorecard readable
    if online:
        from store_support.evals.online import run_online

        card = run_online()
    else:
        card = run_offline()
    print_scorecard(card)
    m = card["metrics"]
    if m["scope_enforcement"] < 1.0 or m["tool_choice_accuracy"] < 1.0:
        print("governance metrics below 1.0 - failing", file=sys.stderr)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
