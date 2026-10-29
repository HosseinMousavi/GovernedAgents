"""Cost accounting. Usage -> USD via a price table; a per-tenant, per-day ledger; budget states.

Prices are USD per million tokens and live in one table so a pricing change is a one-line diff.
Unknown models use a conservative (Sonnet-level) price rather than being free.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Literal

import structlog

log = structlog.get_logger("cost")


@dataclass(frozen=True, slots=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float
    cache_read_per_mtok: float


PRICES: dict[str, Price] = {
    "claude-haiku-4-5": Price(1.00, 5.00, 1.25, 0.10),
    "claude-sonnet-4-5": Price(3.00, 15.00, 3.75, 0.30),
    "claude-opus-4-1": Price(15.00, 75.00, 18.75, 1.50),
}
_DEFAULT_PRICE = PRICES["claude-sonnet-4-5"]


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_write_tokens + self.cache_read_tokens


def price_for(model: str) -> Price:
    for name, p in PRICES.items():
        if model.startswith(name):
            return p
    return _DEFAULT_PRICE


def estimate_cost(model: str, usage: Usage) -> float:
    p = price_for(model)
    return (
        usage.input_tokens * p.input_per_mtok
        + usage.output_tokens * p.output_per_mtok
        + usage.cache_write_tokens * p.cache_write_per_mtok
        + usage.cache_read_tokens * p.cache_read_per_mtok
    ) / 1_000_000


class CostLedger:
    """In-memory per-tenant daily spend + a structured `llm_call` log line per record
    (the durable copy is the log stream; CloudWatch Insights can sum `cost_usd` by tenant)."""

    def __init__(self, today: Callable[[], date] = date.today) -> None:
        self._today = today
        self._day = today()
        self._spent: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def _roll(self) -> None:
        if self._today() != self._day:
            self._day = self._today()
            self._spent = defaultdict(float)

    def record(self, tenant_id: str, cost_usd: float, *, model: str, task: str, usage: Usage | None = None) -> None:
        with self._lock:
            self._roll()
            self._spent[tenant_id] += cost_usd
        log.info(
            "llm_call",
            tenant=tenant_id,
            model=model,
            task=task,
            cost_usd=round(cost_usd, 6),
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            cache_read_tokens=usage.cache_read_tokens if usage else None,
        )

    def spent_today(self, tenant_id: str) -> float:
        with self._lock:
            self._roll()
            return self._spent.get(tenant_id, 0.0)


BudgetState = Literal["ok", "degraded", "exhausted"]


@dataclass(frozen=True, slots=True)
class Budget:
    """Per-tenant daily budget. soft: route everything to the cheap tier. hard: refuse new LLM work."""

    soft_usd: float
    hard_usd: float

    def state(self, spent: float) -> BudgetState:
        if spent >= self.hard_usd:
            return "exhausted"
        if spent >= self.soft_usd:
            return "degraded"
        return "ok"

    def remaining(self, spent: float) -> float:
        return max(0.0, self.soft_usd - spent)
