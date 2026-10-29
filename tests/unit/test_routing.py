"""Model routing is a pure function: (task, signals, policy) -> which model tier runs it.

Policy "routed" (default): cheap models for structured helper work (grade / rewrite / filter),
the default model for the customer-facing answer, the strong model only when the first attempt
struggled. Budget pressure degrades everything to the cheap tier before anything is refused.
"""

from __future__ import annotations

import pytest

from store_support.governance.routing import ModelTiers, RoutingSignals, Task, Tier, route

TIERS = ModelTiers(cheap="claude-haiku-4-5", default="claude-sonnet-4-5", strong="claude-opus-4-1")


@pytest.mark.parametrize("task", [Task.GRADE, Task.REWRITE, Task.FILTER])
def test_helper_tasks_go_to_cheap_tier(task: Task) -> None:
    d = route(RoutingSignals(task=task), TIERS)
    assert d.tier == Tier.CHEAP and d.model == "claude-haiku-4-5"


def test_answer_goes_to_default_tier_on_first_attempt() -> None:
    d = route(RoutingSignals(task=Task.ANSWER, attempt=0, last_score=0.9), TIERS)
    assert d.tier == Tier.DEFAULT and d.model == "claude-sonnet-4-5"


def test_answer_upgrades_to_strong_after_a_retry() -> None:
    d = route(RoutingSignals(task=Task.ANSWER, attempt=1, last_score=0.7), TIERS)
    assert d.tier == Tier.STRONG
    assert "retry" in d.reason


def test_answer_upgrades_to_strong_on_borderline_grade() -> None:
    d = route(RoutingSignals(task=Task.ANSWER, attempt=0, last_score=0.62), TIERS)
    assert d.tier == Tier.STRONG
    assert "borderline" in d.reason


def test_budget_pressure_degrades_everything_to_cheap() -> None:
    d = route(RoutingSignals(task=Task.ANSWER, attempt=1, last_score=0.6, budget_remaining_usd=0.0), TIERS)
    assert d.tier == Tier.CHEAP and d.degraded is True


def test_policy_all_default_ignores_signals() -> None:
    for task in Task:
        d = route(RoutingSignals(task=task, attempt=2, last_score=0.1), TIERS, policy="all_default")
        assert d.tier == Tier.DEFAULT


def test_policy_all_cheap() -> None:
    assert route(RoutingSignals(task=Task.ANSWER), TIERS, policy="all_cheap").tier == Tier.CHEAP


def test_unknown_policy_rejected() -> None:
    with pytest.raises(ValueError, match="policy"):
        route(RoutingSignals(task=Task.ANSWER), TIERS, policy="yolo")


def test_tiers_from_manifest_dict_falls_back_to_single_model() -> None:
    t = ModelTiers.from_manifest({"model": "claude-sonnet-4-5"})
    assert t == ModelTiers(cheap="claude-sonnet-4-5", default="claude-sonnet-4-5", strong="claude-sonnet-4-5")
    t2 = ModelTiers.from_manifest({"model": "m", "models": {"cheap": "h", "strong": "o"}})
    assert t2 == ModelTiers(cheap="h", default="m", strong="o")
