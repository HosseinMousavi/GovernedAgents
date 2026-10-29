"""Cost accounting: token usage -> USD via a price table; a per-tenant daily ledger; budget state."""

from __future__ import annotations

from datetime import date

import pytest

from store_support.governance.cost import PRICES, Budget, CostLedger, Usage, estimate_cost


def test_usage_arithmetic() -> None:
    u = Usage(input_tokens=100, output_tokens=20) + Usage(input_tokens=50, output_tokens=5, cache_read_tokens=10)
    assert u == Usage(input_tokens=150, output_tokens=25, cache_read_tokens=10)
    assert u.total_tokens == 185


def test_estimate_cost_uses_price_table() -> None:
    cost = estimate_cost("claude-sonnet-4-5", Usage(input_tokens=1_000_000, output_tokens=0))
    assert cost == pytest.approx(PRICES["claude-sonnet-4-5"].input_per_mtok)


def test_cache_reads_are_cheaper_than_fresh_input() -> None:
    fresh = estimate_cost("claude-sonnet-4-5", Usage(input_tokens=10_000))
    cached = estimate_cost("claude-sonnet-4-5", Usage(cache_read_tokens=10_000))
    assert cached < fresh * 0.2


def test_haiku_is_cheaper_than_sonnet_for_same_usage() -> None:
    u = Usage(input_tokens=5_000, output_tokens=500)
    assert estimate_cost("claude-haiku-4-5", u) < estimate_cost("claude-sonnet-4-5", u)


def test_unknown_model_uses_conservative_default_price() -> None:
    assert estimate_cost("some-future-model", Usage(input_tokens=1_000_000)) > 0


def test_ledger_is_per_tenant_per_day() -> None:
    led = CostLedger(today=lambda: date(2026, 10, 28))
    led.record("store-a", 0.10, model="m", task="answer")
    led.record("store-a", 0.05, model="m", task="grade")
    led.record("store-b", 1.00, model="m", task="answer")
    assert led.spent_today("store-a") == pytest.approx(0.15)
    assert led.spent_today("store-b") == pytest.approx(1.00)
    assert led.spent_today("store-c") == 0.0


def test_ledger_rolls_over_at_midnight() -> None:
    day = [date(2026, 10, 28)]
    led = CostLedger(today=lambda: day[0])
    led.record("store-a", 0.5, model="m", task="answer")
    day[0] = date(2026, 10, 29)
    assert led.spent_today("store-a") == 0.0


def test_budget_states() -> None:
    b = Budget(soft_usd=1.0, hard_usd=2.0)
    assert b.state(0.2) == "ok"
    assert b.state(1.0) == "degraded"  # soft limit hit: route everything cheap
    assert b.state(2.0) == "exhausted"  # hard limit hit: refuse new LLM work
    assert b.remaining(0.4) == pytest.approx(0.6)
