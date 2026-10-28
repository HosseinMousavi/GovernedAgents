"""Minimal LLM interface: three operations the knowledge agent needs, nothing more.
Anthropic in production, FakeLLM (scripted) in tests and offline evals."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol

from store_support.knowledge.store import Hit

GRADER_SYSTEM = (
    "You grade whether retrieved context can answer a customer question. "
    'Reply with JSON only: {"score": <0..1>, "why": "<10 words>"}. '
    "1.0 = context fully answers it; 0.0 = unrelated."
)
REWRITE_SYSTEM = (
    "Rewrite the customer's question as a short search query for a store policy knowledge base. "
    "Use policy vocabulary (return, refund, warranty, shipping, damaged, cancellation). Reply with the query only."
)
ANSWER_SYSTEM = (
    "You are a store support assistant. Answer ONLY from the provided context. "
    "If the context does not answer the question, reply exactly: INSUFFICIENT_CONTEXT. "
    "Cite every claim with the chunk tag in square brackets exactly as given, e.g. [return-policy.md#Refund timing]. "
    "Be brief and concrete. Never ask for or repeat personal data."
)


class LLM(Protocol):
    def grade(self, question: str, hits: Sequence[Hit]) -> float: ...
    def rewrite(self, question: str, previous_queries: Sequence[str]) -> str: ...
    def answer(self, question: str, hits: Sequence[Hit]) -> str: ...


def _context_block(hits: Sequence[Hit]) -> str:
    return "\n\n".join(f"{h.citation}\n{h.content}" for h in hits)


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, max_tokens: int = 600) -> None:
        from anthropic import Anthropic

        self._c = Anthropic(api_key=api_key)
        self._model = model
        self._max = max_tokens

    def _call(self, system: str, user: str) -> str:
        msg = self._c.messages.create(
            model=self._model, max_tokens=self._max, system=system, messages=[{"role": "user", "content": user}]
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()

    def grade(self, question: str, hits: Sequence[Hit]) -> float:
        raw = self._call(GRADER_SYSTEM, f"Question: {question}\n\nContext:\n{_context_block(hits)}")
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            return max(0.0, min(1.0, float(json.loads(m.group(0))["score"]))) if m else 0.0
        except (ValueError, KeyError, TypeError):
            return 0.0

    def rewrite(self, question: str, previous_queries: Sequence[str]) -> str:
        prev = "\n".join(f"- {q}" for q in previous_queries)
        return self._call(REWRITE_SYSTEM, f"Question: {question}\nAlready tried (avoid repeating):\n{prev}")

    def answer(self, question: str, hits: Sequence[Hit]) -> str:
        return self._call(ANSWER_SYSTEM, f"Context:\n{_context_block(hits)}\n\nQuestion: {question}")


class FakeLLM:
    """Scripted responses; counts calls so tests can assert the loop's shape."""

    def __init__(
        self, *, grades: Sequence[float] = (), rewrites: Sequence[str] = (), answer: str = "INSUFFICIENT_CONTEXT"
    ) -> None:
        self._grades = list(grades)
        self._rewrites = list(rewrites)
        self._answer = answer
        self.grade_calls = self.rewrite_calls = self.answer_calls = 0

    def grade(self, question: str, hits: Sequence[Hit]) -> float:
        self.grade_calls += 1
        return self._grades.pop(0) if self._grades else 0.0

    def rewrite(self, question: str, previous_queries: Sequence[str]) -> str:
        self.rewrite_calls += 1
        return self._rewrites.pop(0) if self._rewrites else f"{question} (rewritten {self.rewrite_calls})"

    def answer(self, question: str, hits: Sequence[Hit]) -> str:
        self.answer_calls += 1
        return self._answer
