"""Claude Agent SDK build: the governance lives in the tool handlers, which we test
directly (no CLI runtime needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from store_support.knowledge.chunking import chunk_markdown
from store_support.knowledge.embeddings import HashEmbedder
from store_support.knowledge.llm import FakeLLM
from store_support.knowledge.pipeline import Outcome
from store_support.knowledge.sdk_agent import RetrievalBudget, SdkKnowledgeAgent
from store_support.knowledge.store import Chunk, InMemoryVectorStore

POLICIES = Path(__file__).resolve().parents[2] / "data" / "policies"


@pytest.fixture
def store() -> InMemoryVectorStore:
    s = InMemoryVectorStore(HashEmbedder(dim=128))
    for f in POLICIES.glob("*.md"):
        s.add("store-a", chunk_markdown(f.read_text(), source=f.name))
    s.add("store-b", [Chunk("b-only.md", "B", "tenant b secret")])
    return s


async def test_search_budget_is_enforced_in_code(store) -> None:
    agent = SdkKnowledgeAgent(store=store, llm=FakeLLM(), tenant_id="store-a", max_retries=2)
    budget = RetrievalBudget(max_searches=3)
    h = agent.handlers(budget)
    for _ in range(3):
        assert "is_error" not in await h["search_policies"]({"query": "refund"})
    out = await h["search_policies"]({"query": "refund again"})
    assert out["is_error"] is True
    assert budget.searches == 3


async def test_tenant_is_bound_not_chosen_by_model(store) -> None:
    agent = SdkKnowledgeAgent(store=store, llm=FakeLLM(), tenant_id="store-a")
    h = agent.handlers(RetrievalBudget(max_searches=3))
    out = await h["search_policies"]({"query": "tenant b secret", "tenant_id": "store-b"})  # extra arg ignored
    assert "b-only.md" not in out["content"][0]["text"]


async def test_grade_uses_shared_grader(store) -> None:
    llm = FakeLLM(grades=[0.75])
    agent = SdkKnowledgeAgent(store=store, llm=llm, tenant_id="store-a")
    budget = RetrievalBudget(max_searches=3)
    h = agent.handlers(budget)
    await h["search_policies"]({"query": "refund"})
    out = await h["grade_context"]({"question": "refund"})
    assert '"score": 0.75' in out["content"][0]["text"]
    assert budget.scores == [0.75]


async def test_finalize_validates_citations_like_langgraph(store) -> None:
    agent = SdkKnowledgeAgent(store=store, llm=FakeLLM(), tenant_id="store-a")
    budget = RetrievalBudget(max_searches=3)
    h = agent.handlers(budget)
    await h["search_policies"]({"query": "refund timing"})
    good = agent.finalize("5-7 business days. [return-policy.md#Refund timing]", budget)
    assert good.outcome == Outcome.ANSWERED
    bad = agent.finalize("5-7 business days, trust me.", budget)
    assert bad.outcome == Outcome.ESCALATED and bad.reason == "no_citation"


async def test_escalate_tool_wins_over_answer_text(store) -> None:
    agent = SdkKnowledgeAgent(store=store, llm=FakeLLM(), tenant_id="store-a")
    budget = RetrievalBudget(max_searches=3)
    h = agent.handlers(budget)
    await h["escalate"]({"reason": "low_relevance"})
    out = agent.finalize("some answer [return-policy.md#Refund timing]", budget)
    assert out.outcome == Outcome.ESCALATED and out.reason == "low_relevance"
