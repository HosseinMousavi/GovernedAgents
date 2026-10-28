r"""Knowledge agent, LangGraph edition.

Explicit state machine: every node is a pipeline step, every edge is a visible rule.

    START -> retrieve -> grade --(ok)--> generate -> END
                            \--(low, retries left)--> rewrite -> retrieve
                             \--(low, exhausted)----> escalate -> END
                   generate --(bad citation)--------> escalate
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from store_support.knowledge import pipeline as p
from store_support.knowledge.llm import LLM
from store_support.knowledge.store import Hit, VectorStore


class State(TypedDict, total=False):
    question: str
    query: str
    queries: list[str]
    hits: list[Hit]
    scores: list[float]
    retries: int
    answer: str
    citations: list[p.Citation]
    outcome: p.Outcome
    reason: str | None


def build_graph(
    *, store: VectorStore, llm: LLM, tenant_id: str, k: int = 5, max_retries: int = 2, threshold: float = 0.6
) -> CompiledStateGraph[Any, Any, Any, Any]:
    def retrieve(s: State) -> State:
        q = s.get("query") or s["question"]
        hits = p.retrieve(store, tenant_id, q, k=k)
        return {"query": q, "queries": [*s.get("queries", []), q], "hits": hits}

    def grade(s: State) -> State:
        if not s["hits"]:
            return {"scores": s.get("scores", []), "reason": "no_context"}
        return {"scores": [*s.get("scores", []), p.grade(llm, s["question"], s["hits"])]}

    def after_grade(s: State) -> Literal["generate", "rewrite", "escalate"]:
        if s.get("reason") == "no_context":
            return "escalate"
        if s["scores"][-1] >= threshold:
            return "generate"
        return "rewrite" if s.get("retries", 0) < max_retries else "escalate"

    def rewrite(s: State) -> State:
        return {"retries": s.get("retries", 0) + 1, "query": p.rewrite(llm, s["question"], s["queries"])}

    def generate(s: State) -> State:
        answer = p.generate(llm, s["question"], s["hits"])
        cites, reason = p.validate_answer(answer, s["hits"])
        if reason:
            return {"answer": answer, "reason": reason}
        return {"answer": answer, "citations": cites, "outcome": p.Outcome.ANSWERED, "reason": None}

    def after_generate(s: State) -> Literal["escalate", "__end__"]:
        return "escalate" if s.get("reason") else END

    def escalate(s: State) -> State:
        return {"outcome": p.Outcome.ESCALATED, "reason": s.get("reason") or "low_relevance"}

    g: StateGraph[State] = StateGraph(State)
    for name, fn in [
        ("retrieve", retrieve),
        ("grade", grade),
        ("rewrite", rewrite),
        ("generate", generate),
        ("escalate", escalate),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade)
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("generate", after_generate)
    g.add_edge("escalate", END)
    return g.compile()


def run_graph(graph: CompiledStateGraph[Any, Any, Any, Any], question: str) -> p.KnowledgeResult:
    s: State = graph.invoke({"question": question, "retries": 0, "queries": [], "scores": []})
    return p.KnowledgeResult(
        outcome=s.get("outcome", p.Outcome.ESCALATED),
        answer=s.get("answer", ""),
        citations=s.get("citations", []),
        reason=s.get("reason"),
        retries=s.get("retries", 0),
        queries=s.get("queries", []),
        scores=s.get("scores", []),
        hits=s.get("hits", []),
    )
