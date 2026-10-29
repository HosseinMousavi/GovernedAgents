"""LLM interface for the knowledge agent: grade, rewrite, filter, answer.

RoutedAnthropicLLM (production)
  * every call is routed to a model tier (governance/routing.py) from task + signals + budget
  * prompt caching: the system prompt and the retrieved-context block carry cache_control, so the
    grade -> filter -> answer sequence on the same context pays for the context once
  * usage and cost are recorded per call (governance/cost.py) and drained per request
  * a hard daily budget raises BudgetExhaustedError before any call is made
FakeLLM (tests) and ExtractiveLLM (offline evals) implement the same Protocol with zero cost.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Protocol

from opentelemetry import trace

from store_support.governance.cost import Budget, CostLedger, Usage, estimate_cost
from store_support.governance.routing import ModelTiers, RoutingSignals, Task, route
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
FILTER_SYSTEM = (
    "You select which retrieved chunks are relevant to answering the customer's question. "
    "Chunks are numbered from 0. Reply with a JSON list of the relevant chunk numbers only, e.g. [0, 3]. "
    "Keep a chunk only if it contains information needed for the answer."
)
ANSWER_SYSTEM = (
    "You are a store support assistant. Answer ONLY from the provided context. "
    "If the context does not answer the question, reply exactly: INSUFFICIENT_CONTEXT. "
    "Cite every claim with the chunk tag in square brackets exactly as given, e.g. [return-policy.md#Refund timing]. "
    "Be brief and concrete. Never ask for or repeat personal data."
)


LOCAL_PREFIX = "ollama/"  # e.g. models.cheap: ollama/ssa-grader (the LoRA-tuned local tier)


class OllamaChat:
    """Minimal /api/chat client for the local tier. Returns (text, usage); usage is informational."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        import httpx

        self._url = base_url.rstrip("/") + "/api/chat"
        self._client = httpx.Client(timeout=timeout)

    def chat(self, *, model: str, system: str, user: str) -> tuple[str, Usage]:
        r = self._client.post(
            self._url,
            json={
                "model": model,
                "stream": False,
                "options": {"temperature": 0},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
        )
        r.raise_for_status()
        body = r.json()
        usage = Usage(input_tokens=int(body.get("prompt_eval_count", 0)), output_tokens=int(body.get("eval_count", 0)))
        return str(body.get("message", {}).get("content", "")), usage


class BudgetExhaustedError(RuntimeError):
    """Daily hard budget for this tenant is spent; the caller should escalate to a human."""


class LLM(Protocol):
    def grade(self, question: str, hits: Sequence[Hit]) -> float: ...
    def rewrite(self, question: str, previous_queries: Sequence[str]) -> str: ...
    def filter(self, question: str, hits: Sequence[Hit]) -> list[Hit]: ...
    def answer(
        self, question: str, hits: Sequence[Hit], *, attempt: int = 0, last_score: float | None = None
    ) -> str: ...
    def drain_usage(self) -> tuple[Usage, float, list[str]]: ...


def _context_block(hits: Sequence[Hit]) -> str:
    return "\n\n".join(f"{h.citation}\n{h.content}" for h in hits)


def _numbered_block(hits: Sequence[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.citation}\n{h.content}" for i, h in enumerate(hits))


def parse_score(raw: str) -> float:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return max(0.0, min(1.0, float(json.loads(m.group(0))["score"]))) if m else 0.0
    except (ValueError, KeyError, TypeError):
        return 0.0


def parse_indices(raw: str, n: int) -> list[int] | None:
    m = re.search(r"\[[\d,\s]*\]", raw)
    if not m:
        return None
    try:
        idx = sorted({int(i) for i in json.loads(m.group(0))})
    except (ValueError, TypeError):
        return None
    return [i for i in idx if 0 <= i < n]


class RoutedAnthropicLLM:
    def __init__(
        self,
        *,
        tiers: ModelTiers,
        tenant_id: str,
        client: Any | None = None,
        ollama: OllamaChat | None = None,
        api_key: str = "",
        policy: str = "routed",
        ledger: CostLedger | None = None,
        budget: Budget | None = None,
        max_tokens: int = 600,
    ) -> None:
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)
        self._c = client
        self._ollama = ollama
        self._tiers = tiers
        self._tenant = tenant_id
        self._policy = policy
        self._ledger = ledger or CostLedger()
        self._budget = budget
        self._max = max_tokens
        self._usage = Usage()
        self._cost = 0.0
        self._models: list[str] = []

    # ----- routing + accounting -----

    def _budget_remaining(self) -> float | None:
        if self._budget is None:
            return None
        spent = self._ledger.spent_today(self._tenant)
        if self._budget.state(spent) == "exhausted":
            raise BudgetExhaustedError(f"tenant daily budget exhausted ({spent:.2f} USD)")
        return self._budget.remaining(spent)

    def _call(
        self,
        task: Task,
        system: str,
        blocks: list[dict[str, Any]],
        *,
        attempt: int = 0,
        last_score: float | None = None,
    ) -> str:
        signals = RoutingSignals(
            task=task,
            attempt=attempt,
            last_score=last_score,
            context_chars=sum(len(b.get("text", "")) for b in blocks),
            budget_remaining_usd=self._budget_remaining(),
        )
        decision = route(signals, self._tiers, self._policy)
        if decision.model.startswith(LOCAL_PREFIX):
            # Local (LoRA) tier: served by Ollama, no per-token price.
            if self._ollama is None:
                raise RuntimeError(f"model {decision.model!r} needs an ollama client (OLLAMA_BASE_URL)")
            text, usage = self._ollama.chat(
                model=decision.model[len(LOCAL_PREFIX) :],
                system=system,
                user="".join(b.get("text", "") for b in blocks),
            )
            cost = 0.0
        else:
            msg = self._c.messages.create(
                model=decision.model,
                max_tokens=self._max,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": blocks}],
            )
            u = getattr(msg, "usage", None)
            usage = Usage(
                input_tokens=int(getattr(u, "input_tokens", 0) or 0),
                output_tokens=int(getattr(u, "output_tokens", 0) or 0),
                cache_write_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
                cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
            cost = estimate_cost(decision.model, usage)
        self._usage += usage
        self._cost += cost
        self._models.append(decision.model)
        self._ledger.record(self._tenant, cost, model=decision.model, task=task.value, usage=usage)
        span = trace.get_current_span()
        span.set_attribute("llm.model", decision.model)
        span.set_attribute("llm.tier", decision.tier.value)
        span.set_attribute("llm.route_reason", decision.reason)
        span.set_attribute("llm.degraded", decision.degraded)
        span.set_attribute("llm.input_tokens", usage.input_tokens)
        span.set_attribute("llm.output_tokens", usage.output_tokens)
        span.set_attribute("llm.cache_read_tokens", usage.cache_read_tokens)
        span.set_attribute("llm.cost_usd", round(cost, 6))
        return text.strip()

    def drain_usage(self) -> tuple[Usage, float, list[str]]:
        out = (self._usage, self._cost, list(self._models))
        self._usage, self._cost, self._models = Usage(), 0.0, []
        return out

    # ----- operations -----

    @staticmethod
    def _ctx(hits: Sequence[Hit], text: str) -> dict[str, Any]:
        return {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}

    def grade(self, question: str, hits: Sequence[Hit]) -> float:
        blocks = [
            self._ctx(hits, f"Context:\n{_context_block(hits)}"),
            {"type": "text", "text": f"\n\nQuestion: {question}"},
        ]
        return parse_score(self._call(Task.GRADE, GRADER_SYSTEM, blocks))

    def rewrite(self, question: str, previous_queries: Sequence[str]) -> str:
        prev = "\n".join(f"- {q}" for q in previous_queries)
        blocks = [{"type": "text", "text": f"Question: {question}\nAlready tried (avoid repeating):\n{prev}"}]
        return self._call(Task.REWRITE, REWRITE_SYSTEM, blocks)

    def filter(self, question: str, hits: Sequence[Hit]) -> list[Hit]:
        if len(hits) <= 1:
            return list(hits)
        blocks = [
            {
                "type": "text",
                "text": f"Chunks:\n{_numbered_block(hits)}\n\nQuestion: {question}\nRelevant chunk numbers:",
            }
        ]
        idx = parse_indices(self._call(Task.FILTER, FILTER_SYSTEM, blocks), len(hits))
        if not idx:  # None (garbage) or [] (model dropped everything): never starve the answer step
            return list(hits)
        return [hits[i] for i in idx]

    def answer(self, question: str, hits: Sequence[Hit], *, attempt: int = 0, last_score: float | None = None) -> str:
        blocks = [
            self._ctx(hits, f"Context:\n{_context_block(hits)}"),
            {"type": "text", "text": f"\n\nQuestion: {question}"},
        ]
        return self._call(Task.ANSWER, ANSWER_SYSTEM, blocks, attempt=attempt, last_score=last_score)


class FakeLLM:
    """Scripted responses; counts calls so tests can assert the loop's shape. Zero cost."""

    def __init__(
        self,
        *,
        grades: Sequence[float] = (),
        rewrites: Sequence[str] = (),
        answer: str = "INSUFFICIENT_CONTEXT",
        keep: Sequence[int] | None = None,
    ) -> None:
        self._grades = list(grades)
        self._rewrites = list(rewrites)
        self._answer = answer
        self._keep = list(keep) if keep is not None else None
        self.grade_calls = self.rewrite_calls = self.answer_calls = self.filter_calls = 0

    def grade(self, question: str, hits: Sequence[Hit]) -> float:
        self.grade_calls += 1
        return self._grades.pop(0) if self._grades else 0.0

    def rewrite(self, question: str, previous_queries: Sequence[str]) -> str:
        self.rewrite_calls += 1
        return self._rewrites.pop(0) if self._rewrites else f"{question} (rewritten {self.rewrite_calls})"

    def filter(self, question: str, hits: Sequence[Hit]) -> list[Hit]:
        self.filter_calls += 1
        if self._keep is None:
            return list(hits)
        return [hits[i] for i in self._keep if i < len(hits)]

    def answer(self, question: str, hits: Sequence[Hit], *, attempt: int = 0, last_score: float | None = None) -> str:
        self.answer_calls += 1
        return self._answer

    def drain_usage(self) -> tuple[Usage, float, list[str]]:
        return Usage(), 0.0, []
