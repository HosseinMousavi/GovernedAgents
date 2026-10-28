"""The LangGraph build must behave exactly like the reference pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from store_support.knowledge.chunking import chunk_markdown
from store_support.knowledge.embeddings import HashEmbedder
from store_support.knowledge.langgraph_agent import build_graph, run_graph
from store_support.knowledge.llm import FakeLLM
from store_support.knowledge.pipeline import Outcome
from store_support.knowledge.store import InMemoryVectorStore

POLICIES = Path(__file__).resolve().parents[2] / "data" / "policies"


@pytest.fixture
def store() -> InMemoryVectorStore:
    s = InMemoryVectorStore(HashEmbedder(dim=128))
    for f in POLICIES.glob("*.md"):
        s.add("store-a", chunk_markdown(f.read_text(), source=f.name))
    return s


def test_graph_has_expected_nodes(store) -> None:
    g = build_graph(store=store, llm=FakeLLM(), tenant_id="store-a")
    assert {"retrieve", "grade", "rewrite", "generate", "escalate"} <= set(g.get_graph().nodes)


def test_graph_answers_on_high_grade(store) -> None:
    llm = FakeLLM(grades=[0.9], answer="Report within 7 days. [manual-steering-wheel-cover.md#Common issues]")
    out = run_graph(build_graph(store=store, llm=llm, tenant_id="store-a"), "steering wheel cover arrived damaged")
    assert out.outcome == Outcome.ANSWERED
    assert out.citations[0].source == "manual-steering-wheel-cover.md"


def test_graph_rewrites_then_escalates_after_two_retries(store) -> None:
    llm = FakeLLM(grades=[0.1, 0.2, 0.3], rewrites=["q2", "q3"])
    out = run_graph(build_graph(store=store, llm=llm, tenant_id="store-a"), "???")
    assert out.outcome == Outcome.ESCALATED
    assert out.retries == 2
    assert out.queries == ["???", "q2", "q3"]
    assert llm.answer_calls == 0


def test_graph_rejects_uncited_answer(store) -> None:
    llm = FakeLLM(grades=[0.9], answer="yes definitely")
    out = run_graph(build_graph(store=store, llm=llm, tenant_id="store-a"), "refund?")
    assert out.outcome == Outcome.ESCALATED
    assert out.reason == "no_citation"
