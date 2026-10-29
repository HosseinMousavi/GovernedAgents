"""RoutedAnthropicLLM: every call picks a model via `route`, uses prompt caching, records
usage + cost in the ledger, and refuses when the hard budget is exhausted.
Anthropic's client is stubbed; no network."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest

from store_support.governance.cost import Budget, CostLedger, Usage, estimate_cost
from store_support.governance.routing import ModelTiers
from store_support.knowledge.llm import BudgetExhaustedError, RoutedAnthropicLLM
from store_support.knowledge.store import Hit

TIERS = ModelTiers(cheap="claude-haiku-4-5", default="claude-sonnet-4-5", strong="claude-opus-4-1")
HITS = [
    Hit("return-policy.md", "Refund timing", "Approved refunds post within 5-7 business days.", 0.9),
    Hit("shipping-policy.md", "Tracking", "A tracking number is emailed when shipped.", 0.5),
    Hit("warranty-policy.md", "Coverage", "Interior accessories: 12 months.", 0.4),
]


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Msg:
    content: list[Any]
    usage: Any


@dataclass
class _Usage:
    input_tokens: int = 1000
    output_tokens: int = 50
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class StubClient:
    """Records every messages.create call; returns scripted text per task keyword."""

    replies: dict[str, str] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def messages(self) -> StubClient:
        return self

    def create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        system_text = kw["system"][0]["text"] if isinstance(kw["system"], list) else kw["system"]
        key = next((k for k in self.replies if k in system_text), "")
        text = self.replies.get(key, "")

        return _Msg(content=[_Block(text)], usage=_Usage())


def _llm(client: StubClient, **kw: Any) -> RoutedAnthropicLLM:
    return RoutedAnthropicLLM(client=client, tiers=TIERS, tenant_id="store-a", **kw)


def test_grade_uses_cheap_model_and_answer_uses_default() -> None:
    c = StubClient(replies={"grade": '{"score": 0.9}', "support assistant": "ok [return-policy.md#Refund timing]"})
    llm = _llm(c)
    assert llm.grade("q", HITS) == 0.9
    llm.answer("q", HITS, attempt=0, last_score=0.9)
    assert [k["model"] for k in c.calls] == ["claude-haiku-4-5", "claude-sonnet-4-5"]


def test_answer_after_retry_uses_strong_model() -> None:
    c = StubClient(replies={"support assistant": "ok"})
    _llm(c).answer("q", HITS, attempt=1, last_score=0.8)
    assert c.calls[-1]["model"] == "claude-opus-4-1"


def test_prompt_caching_markers_on_system_and_context() -> None:
    c = StubClient(replies={"support assistant": "ok"})
    _llm(c).answer("q", HITS, attempt=0, last_score=0.9)
    call = c.calls[-1]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    user_blocks = call["messages"][0]["content"]
    assert any(b.get("cache_control") == {"type": "ephemeral"} for b in user_blocks)  # context block
    assert "Refund timing" in "".join(b["text"] for b in user_blocks)


def test_filter_keeps_only_relevant_hits_and_runs_cheap() -> None:
    c = StubClient(replies={"relevant": "[0, 2]"})
    kept = _llm(c).filter("q", HITS)
    assert [h.source for h in kept] == ["return-policy.md", "warranty-policy.md"]
    assert c.calls[-1]["model"] == "claude-haiku-4-5"


def test_filter_falls_back_to_all_hits_on_garbage_output() -> None:
    c = StubClient(replies={"relevant": "I think the first one"})
    assert _llm(c).filter("q", HITS) == HITS


def test_usage_and_cost_are_recorded_per_call_and_drained_per_request() -> None:
    led = CostLedger(today=lambda: date(2026, 10, 28))
    c = StubClient(replies={"grade": '{"score": 0.9}', "support assistant": "ok"})
    llm = _llm(c, ledger=led)
    llm.grade("q", HITS)
    llm.answer("q", HITS, attempt=0, last_score=0.9)
    usage, cost, models = llm.drain_usage()
    assert usage == Usage(input_tokens=2000, output_tokens=100)
    assert cost > 0 and models == ["claude-haiku-4-5", "claude-sonnet-4-5"]
    assert led.spent_today("store-a") == pytest.approx(cost)
    assert llm.drain_usage()[0] == Usage()  # drained


def test_soft_budget_degrades_to_cheap_and_hard_budget_refuses() -> None:
    led = CostLedger(today=lambda: date(2026, 10, 28))
    led.record("store-a", 1.0, model="m", task="x")
    c = StubClient(replies={"support assistant": "ok"})
    llm = _llm(c, ledger=led, budget=Budget(soft_usd=1.0, hard_usd=2.0))
    llm.answer("q", HITS, attempt=1, last_score=0.5)  # would be strong, but degraded
    assert c.calls[-1]["model"] == "claude-haiku-4-5"
    led.record("store-a", 1.0, model="m", task="x")
    with pytest.raises(BudgetExhaustedError):
        llm.answer("q", HITS)


def test_policy_all_default_runs_everything_on_default() -> None:
    c = StubClient(replies={"grade": '{"score": 0.9}', "support assistant": "ok"})
    llm = _llm(c, policy="all_default")
    llm.grade("q", HITS)
    llm.answer("q", HITS, attempt=1)
    assert {k["model"] for k in c.calls} == {"claude-sonnet-4-5"}


# ---------- local (LoRA) tier: ollama/<model> is dispatched to Ollama and costs $0 ----------


@dataclass
class StubOllama:
    replies: dict[str, str] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat(self, *, model: str, system: str, user: str) -> tuple[str, Usage]:
        self.calls.append({"model": model, "system": system, "user": user})
        key = next((k for k in self.replies if k in system), "")
        return self.replies.get(key, ""), Usage(input_tokens=300, output_tokens=10)


def test_cheap_tier_can_be_a_local_ollama_model() -> None:
    tiers = ModelTiers(cheap="ollama/ssa-grader", default="claude-sonnet-4-5", strong="claude-opus-4-1")
    anthropic = StubClient(replies={"support assistant": "ok"})
    local = StubOllama(replies={"grade": '{"score": 0.7}', "relevant": "[1]"})
    llm = RoutedAnthropicLLM(client=anthropic, ollama=local, tiers=tiers, tenant_id="store-a")
    assert llm.grade("q", HITS) == 0.7
    assert [h.source for h in llm.filter("q", HITS)] == ["shipping-policy.md"]
    llm.answer("q", HITS, attempt=0, last_score=0.9)
    assert [c["model"] for c in local.calls] == ["ssa-grader", "ssa-grader"]
    assert [c["model"] for c in anthropic.calls] == ["claude-sonnet-4-5"]
    usage, cost, models = llm.drain_usage()
    assert models == ["ollama/ssa-grader", "ollama/ssa-grader", "claude-sonnet-4-5"]
    assert cost == pytest.approx(estimate_cost("claude-sonnet-4-5", Usage(input_tokens=1000, output_tokens=50)))


def test_local_model_without_ollama_client_is_a_configuration_error() -> None:
    tiers = ModelTiers(cheap="ollama/ssa-grader", default="claude-sonnet-4-5", strong="claude-opus-4-1")
    llm = RoutedAnthropicLLM(client=StubClient(), tiers=tiers, tenant_id="store-a")
    with pytest.raises(RuntimeError, match="ollama"):
        llm.grade("q", HITS)
