"""In-memory repository for unit tests and offline evals. Same contract as Postgres."""

from __future__ import annotations

import itertools
from dataclasses import replace
from datetime import date

from store_support.access import RowFilter
from store_support.mcp_server.repo import Inventory, Order


class InMemoryRepository:
    def __init__(self) -> None:
        self._customers: dict[str, dict[str, str]] = {}
        self._products: dict[tuple[str, str], Inventory] = {}
        self._orders: dict[int, Order] = {}
        self.tickets: list[dict[str, object]] = []
        self.appointments: list[dict[str, object]] = []
        self.refunds: list[dict[str, object]] = []
        self._seq = itertools.count(1)

    # --- seeding helpers (test-only) ---
    def add_customer(self, customer_id: str, *, tenant_id: str, email: str) -> None:
        self._customers[customer_id] = {"tenant_id": tenant_id, "email": email}

    def add_product(self, sku: str, *, tenant_id: str, name: str, price_cents: int, stock: int) -> None:
        self._products[(tenant_id, sku)] = Inventory(sku, tenant_id, name, stock, price_cents)

    def add_order(
        self,
        order_id: int,
        *,
        customer_id: str,
        tenant_id: str,
        sku: str,
        total_cents: int,
        warranty_months: int,
        status: str = "delivered",
        ordered_on: date | None = None,
    ) -> None:
        prod = self._products.get((tenant_id, sku))
        self._orders[order_id] = Order(
            order_id=order_id,
            tenant_id=tenant_id,
            customer_id=customer_id,
            sku=sku,
            product_name=prod.product_name if prod else sku,
            total_cents=total_cents,
            status=status,
            ordered_on=ordered_on or date(2026, 10, 1),
            warranty_months=warranty_months,
        )

    # --- Repository protocol ---
    def get_order(self, order_id: int, filt: RowFilter) -> Order | None:
        o = self._orders.get(order_id)
        if o is None or o.tenant_id != filt.tenant_id:
            return None
        if filt.owner_id is not None and o.customer_id != filt.owner_id:
            return None
        return o

    def get_inventory(self, sku: str, tenant_id: str) -> Inventory | None:
        return self._products.get((tenant_id, sku))

    def create_ticket(self, *, order: Order, category: str, summary: str, created_by: str) -> str:
        tid = f"TCK-{next(self._seq):05d}"
        self.tickets.append(
            {
                "ticket_id": tid,
                "order_id": order.order_id,
                "tenant_id": order.tenant_id,
                "category": category,
                "summary": summary,
                "created_by": created_by,
                "status": "open",
            }
        )
        return tid

    def schedule_service(self, *, order: Order, preferred_date: date, created_by: str) -> str:
        aid = f"APT-{next(self._seq):05d}"
        self.appointments.append(
            {
                "appointment_id": aid,
                "order_id": order.order_id,
                "tenant_id": order.tenant_id,
                "date": preferred_date.isoformat(),
                "created_by": created_by,
            }
        )
        return aid

    def issue_refund(self, *, order: Order, amount_cents: int, reason: str, approved_by: str) -> str:
        rid = f"RFD-{next(self._seq):05d}"
        self.refunds.append(
            {
                "refund_id": rid,
                "order_id": order.order_id,
                "tenant_id": order.tenant_id,
                "amount_cents": amount_cents,
                "reason": reason,
                "approved_by": approved_by,
            }
        )
        self._orders[order.order_id] = replace(
            order, status="refunded", refunded_cents=order.refunded_cents + amount_cents
        )
        return rid
