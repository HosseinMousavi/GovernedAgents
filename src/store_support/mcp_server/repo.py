"""Repository protocol. Every read takes a RowFilter so tenant/owner constraints are
enforced at the data layer, not left to the caller's memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from store_support.access import RowFilter


@dataclass(frozen=True, slots=True)
class Order:
    order_id: int
    tenant_id: str
    customer_id: str
    sku: str
    product_name: str
    total_cents: int
    status: str
    ordered_on: date
    warranty_months: int
    refunded_cents: int = 0


@dataclass(frozen=True, slots=True)
class Inventory:
    sku: str
    tenant_id: str
    product_name: str
    stock: int
    price_cents: int


class RefundConflictError(Exception):
    """Raised by issue_refund when the order was already refunded (atomic check-and-set)."""


class Repository(Protocol):
    def get_order(self, order_id: int, filt: RowFilter) -> Order | None: ...
    def get_inventory(self, sku: str, tenant_id: str) -> Inventory | None: ...
    def create_ticket(self, *, order: Order, category: str, summary: str, created_by: str) -> str: ...
    def schedule_service(self, *, order: Order, preferred_date: date, created_by: str) -> str: ...
    def issue_refund(self, *, order: Order, amount_cents: int, reason: str, approved_by: str) -> str:
        """Must be atomic: refuse (RefundConflictError) if refunded_cents != 0 at write time."""
        ...
