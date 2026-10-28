"""Knowledge agent: RAG over a vector store, LLM grader, rewrite on low score, 2 retries,
then human escalation. Answers come only from context, with citations."""

from __future__ import annotations

from pathlib import Path

import pytest

from store_support.knowledge.chunking import chunk_markdown
from store_support.knowledge.embeddings import FallbackEmbedder, HashEmbedder
from store_support.knowledge.llm import FakeLLM
from store_support.knowledge.pipeline import KnowledgeAgent, Outcome
from store_support.knowledge.store import Chunk, InMemoryVectorStore

POLICIES = Path(__file__).resolve().parents[2] / "data" / "policies"


# ---------- chunking ----------


def test_chunk_markdown_splits_on_h2_and_keeps_source() -> None:
    chunks = chunk_markdown((POLICIES / "return-policy.md").read_text(), source="return-policy.md")
    assert {c.section for c in chunks} >= {"Eligibility window", "Damaged or defective on arrival"}
    assert all(c.source == "return-policy.md" for c in chunks)
    assert all(c.content.strip() for c in chunks)


def test_chunk_without_headings_is_one_chunk() -> None:
    assert len(chunk_markdown("# T\n\njust a paragraph", source="x.md")) == 1


# ---------- embeddings ----------


def test_hash_embedder_is_deterministic_and_unit_norm() -> None:
    e = HashEmbedder(dim=64)
    a, b = e.embed(["hello"])[0], e.embed(["hello"])[0]
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6


def test_fallback_embedder_uses_secondary_on_failure() -> None:
    class Broken:
        dim = 64
        name = "broken"

        def embed(self, texts):  # noqa: ANN001, ANN202
            raise ConnectionError("ollama down")

    fb = FallbackEmbedder(primary=Broken(), fallback=HashEmbedder(dim=64))
    assert len(fb.embed(["x"])[0]) == 64
    assert fb.last_used == "hash"


# ---------- store ----------


@pytest.fixture
def store() -> InMemoryVectorStore:
    emb = HashEmbedder(dim=128)
    s = InMemoryVectorStore(emb)
    for f in POLICIES.glob("*.md"):
        s.add("store-a", chunk_markdown(f.read_text(), source=f.name))
    s.add("store-b", [Chunk(source="secret-b.md", section="B only", content="store-b exclusive discount code")])
    return s


def test_search_is_tenant_isolated(store: InMemoryVectorStore) -> None:
    hits = store.search("store-a", "store-b exclusive discount code", k=20)
    assert all(h.source != "secret-b.md" for h in hits)
    assert store.search("store-b", "discount code", k=1)[0].source == "secret-b.md"


# ---------- agent loop ----------


def _agent(store: InMemoryVectorStore, llm: FakeLLM) -> KnowledgeAgent:
    return KnowledgeAgent(store=store, llm=llm, tenant_id="store-a", k=4, max_retries=2, threshold=0.6)


def test_answers_with_citation_when_grade_is_high(store: InMemoryVectorStore) -> None:
    llm = FakeLLM(grades=[0.9], answer="Report it within 7 days for a free replacement or full refund. [manual-steering-wheel-cover.md#Common issues]")
    out = _agent(store, llm).run("my steering wheel cover arrived damaged, what can I do?")
    assert out.outcome == Outcome.ANSWERED
    assert out.citations and out.citations[0].source == "manual-steering-wheel-cover.md"
    assert out.retries == 0


def test_rewrites_and_retries_on_low_grade_then_succeeds(store: InMemoryVectorStore) -> None:
    llm = FakeLLM(grades=[0.2, 0.8], rewrites=["damaged on arrival refund policy"], answer="Yes. [return-policy.md#Damaged or defective on arrival]")
    out = _agent(store, llm).run("thing broke??")
    assert out.outcome == Outcome.ANSWERED
    assert out.retries == 1
    assert out.queries == ["thing broke??", "damaged on arrival refund policy"]


def test_escalates_after_two_retries(store: InMemoryVectorStore) -> None:
    llm = FakeLLM(grades=[0.1, 0.1, 0.1], rewrites=["q2", "q3"])
    out = _agent(store, llm).run("unrelated question about quantum physics")
    assert out.outcome == Outcome.ESCALATED
    assert out.retries == 2
    assert llm.answer_calls == 0  # never asked to answer without good context


def test_answer_without_citation_is_rejected(store: InMemoryVectorStore) -> None:
    """A generated answer that cites nothing is treated as unsupported -> escalate."""
    llm = FakeLLM(grades=[0.9], answer="Sure, you get a refund, trust me.")
    out = _agent(store, llm).run("refund?")
    assert out.outcome == Outcome.ESCALATED
    assert out.reason == "no_citation"


def test_citation_must_match_retrieved_source(store: InMemoryVectorStore) -> None:
    llm = FakeLLM(grades=[0.9], answer="Refund ok. [made-up.md#Nope]")
    out = _agent(store, llm).run("refund?")
    assert out.outcome == Outcome.ESCALATED
    assert out.reason == "unknown_citation"


def test_empty_store_escalates_without_calling_llm() -> None:
    empty = InMemoryVectorStore(HashEmbedder(dim=32))
    llm = FakeLLM(grades=[])
    out = KnowledgeAgent(store=empty, llm=llm, tenant_id="store-a").run("anything")
    assert out.outcome == Outcome.ESCALATED
    assert out.reason == "no_context"
    assert llm.grade_calls == 0
