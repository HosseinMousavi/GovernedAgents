"""Resolution rules. Pure: (spec, order facts, today) -> Decision. No I/O, no LLM.

These encode the policies in data/policies/*.md so the proposal is deterministic and
auditable; the LLM's job is to explain the policy to the customer, not to decide money.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any

from store_support.intake.schema import IssueSpec

DAMAGE_REPORT_WINDOW_DAYS = 7
RETURN_WINDOW_DAYS = 30


class Action(StrEnum):
    PROPOSE_REFUND = "propose_refund"          # needs manager:refund approval in n8n
    PROPOSE_REPLACEMENT = "propose_replacement"
    TICKET_ONLY = "ticket_only"
    ANSWER_ONLY = "answer_only"
    ESCALATE = "escalate"


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    amount_cents: int
    requires_manager: bool
    rationale: str


def _days_since(iso: str, today: date) -> int:
    return (today - date.fromisoformat(iso)).days


def decide(spec: IssueSpec, order: dict[str, Any], *, today: date, in_stock: bool | None = None) -> Decision:
    if spec.desired_outcome == "information":
        return Decision(Action.ANSWER_ONLY, 0, False, "customer asked for information only")

    if order["status"] != "delivered":
        return Decision(Action.TICKET_ONLY, 0, False, f"order status is {order['status']}; nothing to refund yet")
    if order.get("refunded_cents", 0) > 0 or order["status"] == "refunded":
        return Decision(Action.TICKET_ONLY, 0, False, "order already refunded; one refund per order")

    age = _days_since(order["ordered_on"], today)
    total = int(order["total_cents"])

    if spec.category in {"damaged", "defective", "wrong_item", "missing"}:
        if age > DAMAGE_REPORT_WINDOW_DAYS:
            return Decision(Action.TICKET_ONLY, 0, False,
                            f"reported {age} days after delivery; damaged-on-arrival window is {DAMAGE_REPORT_WINDOW_DAYS} days")
        if spec.desired_outcome == "replacement" and in_stock:
            return Decision(Action.PROPOSE_REPLACEMENT, 0, False, "damaged on arrival within window; replacement in stock")
        if spec.desired_outcome in {"refund", "replacement"}:
            return Decision(Action.PROPOSE_REFUND, total, True,
                            "damaged on arrival within window; full refund requires manager approval")
        return Decision(Action.TICKET_ONLY, 0, False, "repair requested; route to staff")

    if age <= RETURN_WINDOW_DAYS and spec.desired_outcome == "refund":
        return Decision(Action.PROPOSE_REFUND, total, True, "within 30-day return window; refund requires manager approval")
    return Decision(Action.TICKET_ONLY, 0, False, "outside standard windows; staff review")
