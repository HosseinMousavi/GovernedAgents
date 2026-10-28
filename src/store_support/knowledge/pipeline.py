"""The knowledge loop as plain, framework-free steps.

    retrieve -> grade -> (score >= threshold ? answer : rewrite -> retrieve ...)
    after `max_retries` rewrites with no good context -> ESCALATE (human)
    answer with no citation / unknown citation / INSUFFICIENT_CONTEXT -> ESCALATE

`KnowledgeAgent.run` is the reference implementation. langgraph_agent.py and sdk_agent.py
orchestrate these same step functions - that is what the README comparison is about.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from store_support.knowledge.llm import LLM
from store_support.knowledge.store import Hit, VectorStore
from store_support.telemetry.tracing import step_span

AGENT_NAME = "knowledge-agent"
_CITE = re.compile(r"\[([^\[\]#]+\.md)#([^\[\]]+)\]")


class Outcome(StrEnum):
    ANSWERED = "answered"
    ESCALATED = "escalated"


@dataclass(frozen=True, slots=True)
class Citation:
    source: str
    section: str


@dataclass(slots=True)
class KnowledgeResult:
    outcome: Outcome
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    reason: str | None = None  # why escalated
    retries: int = 0
    queries: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)


# ---------- pure steps ----------


def retrieve(store: VectorStore, tenant_id: str, query: str, *, k: int) -> list[Hit]:
    with step_span("retrieve", agent=AGENT_NAME, attributes={"query": query, "k": k}):
        return store.search(tenant_id, query, k=k)


def grade(llm: LLM, question: str, hits: Sequence[Hit]) -> float:
    with step_span("grade", agent=AGENT_NAME, attributes={"n_hits": len(hits)}) as s:
        score = llm.grade(question, hits)
        s.set_attribute("score", score)
        return score


def rewrite(llm: LLM, question: str, previous: Sequence[str]) -> str:
    with step_span("rewrite", agent=AGENT_NAME, attributes={"attempt": len(previous)}):
        return llm.rewrite(question, previous).strip()


def generate(llm: LLM, question: str, hits: Sequence[Hit]) -> str:
    with step_span("generate", agent=AGENT_NAME):
        return llm.answer(question, hits).strip()


def validate_answer(answer: str, hits: Sequence[Hit]) -> tuple[list[Citation], str | None]:
    """Returns (citations, escalation_reason). Reason is None when the answer is acceptable."""
    if not answer or answer.upper().startswith("INSUFFICIENT_CONTEXT"):
        return [], "insufficient_context"
    cites = [Citation(s, sec.strip()) for s, sec in _CITE.findall(answer)]
    if not cites:
        return [], "no_citation"
    known = {(h.source, h.section) for h in hits}
    known_sources = {h.source for h in hits}
    for c in cites:
        if (c.source, c.section) not in known and c.source not in known_sources:
            return cites, "unknown_citation"
    return cites, None


# ---------- reference orchestration ----------


@dataclass(slots=True)
class KnowledgeAgent:
    store: VectorStore
    llm: LLM
    tenant_id: str
    k: int = 5
    max_retries: int = 2
    threshold: float = 0.6

    def run(self, question: str) -> KnowledgeResult:
        res = KnowledgeResult(outcome=Outcome.ESCALATED)
        query = question
        while True:
            res.queries.append(query)
            hits = retrieve(self.store, self.tenant_id, query, k=self.k)
            if not hits:
                res.reason = "no_context"
                return res
            score = grade(self.llm, question, hits)
            res.scores.append(score)
            if score >= self.threshold:
                res.hits = hits
                answer = generate(self.llm, question, hits)
                cites, reason = validate_answer(answer, hits)
                if reason:
                    res.reason = reason
                    return res
                res.outcome, res.answer, res.citations = Outcome.ANSWERED, answer, cites
                return res
            if res.retries >= self.max_retries:
                res.reason = "low_relevance"
                return res
            res.retries += 1
            query = rewrite(self.llm, question, res.queries)
