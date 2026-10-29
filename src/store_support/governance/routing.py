"""Model routing. Pure: (signals, tiers, policy) -> RouteDecision. No I/O, no globals.

Policies
  routed       grade/rewrite/filter -> cheap; answer -> default; answer after a retry or a
               borderline grade -> strong. Budget "degraded" forces cheap everywhere.
  all_default  every task on the default model (the baseline for the cost comparison)
  all_cheap    every task on the cheap model (the floor for the cost comparison)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

BORDERLINE_GRADE = 0.7  # grade >= threshold (0.6) but below this -> context is thin, spend more on the answer

POLICIES = ("routed", "all_default", "all_cheap")


class Task(StrEnum):
    GRADE = "grade"
    REWRITE = "rewrite"
    FILTER = "filter"
    ANSWER = "answer"


class Tier(StrEnum):
    CHEAP = "cheap"
    DEFAULT = "default"
    STRONG = "strong"


@dataclass(frozen=True, slots=True)
class ModelTiers:
    cheap: str
    default: str
    strong: str

    def model_for(self, tier: Tier) -> str:
        return {Tier.CHEAP: self.cheap, Tier.DEFAULT: self.default, Tier.STRONG: self.strong}[tier]

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> ModelTiers:
        """`model` is the default; `models: {cheap, strong}` are optional overrides."""
        default = str(data["model"])
        models = data.get("models") or {}
        return cls(cheap=str(models.get("cheap", default)), default=default, strong=str(models.get("strong", default)))


@dataclass(frozen=True, slots=True)
class RoutingSignals:
    task: Task
    attempt: int = 0  # 0 = first retrieval, 1+ = after query rewrites
    last_score: float | None = None  # grader score for the context being answered from
    context_chars: int = 0
    budget_remaining_usd: float | None = None  # None = no budget configured


@dataclass(frozen=True, slots=True)
class RouteDecision:
    tier: Tier
    model: str
    reason: str
    degraded: bool = False


def route(signals: RoutingSignals, tiers: ModelTiers, policy: str = "routed") -> RouteDecision:
    if policy not in POLICIES:
        raise ValueError(f"unknown routing policy {policy!r}; expected one of {POLICIES}")
    if policy == "all_default":
        return RouteDecision(Tier.DEFAULT, tiers.default, "policy=all_default")
    if policy == "all_cheap":
        return RouteDecision(Tier.CHEAP, tiers.cheap, "policy=all_cheap")

    if signals.budget_remaining_usd is not None and signals.budget_remaining_usd <= 0:
        return RouteDecision(Tier.CHEAP, tiers.cheap, "budget soft limit reached", degraded=True)

    if signals.task is not Task.ANSWER:
        return RouteDecision(Tier.CHEAP, tiers.cheap, f"{signals.task.value} is structured helper work")

    if signals.attempt >= 1:
        return RouteDecision(Tier.STRONG, tiers.strong, "answer after retry: harder question")
    if signals.last_score is not None and signals.last_score < BORDERLINE_GRADE:
        return RouteDecision(Tier.STRONG, tiers.strong, "answer on borderline context grade")
    return RouteDecision(Tier.DEFAULT, tiers.default, "answer on well-graded context")
