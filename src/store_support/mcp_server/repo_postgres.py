"""Postgres repository. Every SELECT is built with RowFilter.sql() so tenant_id (and
customer_id for read_own) are always in the WHERE clause. PII columns are never selected."""

from __future__ import annotations

import uuid
from datetime import date

from psycopg import Connection
from psycopg.rows import dict_row

from store_support.access import RowFilter
from store_support.mcp_server.repo import Inventory, Order, RefundConflictError


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


class PostgresRepository:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn
        self._conn.row_factory = dict_row  # type: ignore[assignment]

    def get_order(self, order_id: int, filt: RowFilter) -> Order | None:
        where, params = filt.sql(tenant_col="o.tenant_id", owner_col="o.customer_id")
        row = self._conn.execute(
            f"""SELECT o.order_id, o.tenant_id, o.customer_id, o.sku, p.name AS product_name,
                       o.total_cents, o.status, o.ordered_on, p.warranty_months, o.refunded_cents
                FROM orders o JOIN products p ON p.tenant_id = o.tenant_id AND p.sku = o.sku
                WHERE o.order_id = %s AND {where}""",  # noqa: S608 - `where` is built from constants
            [order_id, *params],
        ).fetchone()
        return Order(**row) if row else None  # type: ignore[arg-type]

    def get_inventory(self, sku: str, tenant_id: str) -> Inventory | None:
        row = self._conn.execute(
            """SELECT p.sku, p.tenant_id, p.name AS product_name, i.stock, p.price_cents
               FROM products p JOIN inventory i ON i.tenant_id = p.tenant_id AND i.sku = p.sku
               WHERE p.tenant_id = %s AND p.sku = %s""",
            [tenant_id, sku],
        ).fetchone()
        return Inventory(**row) if row else None  # type: ignore[arg-type]

    def create_ticket(self, *, order: Order, category: str, summary: str, created_by: str) -> str:
        tid = _id("TCK")
        self._conn.execute(
            "INSERT INTO tickets (ticket_id, tenant_id, order_id, category, summary, created_by) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            [tid, order.tenant_id, order.order_id, category, summary, created_by],
        )
        self._conn.commit()
        return tid

    def schedule_service(self, *, order: Order, preferred_date: date, created_by: str) -> str:
        aid = _id("APT")
        self._conn.execute(
            "INSERT INTO appointments (appointment_id, tenant_id, order_id, service_date, created_by) "
            "VALUES (%s,%s,%s,%s,%s)",
            [aid, order.tenant_id, order.order_id, preferred_date, created_by],
        )
        self._conn.commit()
        return aid

    def issue_refund(self, *, order: Order, amount_cents: int, reason: str, approved_by: str) -> str:
        """Atomic check-and-set: the UPDATE only matches when refunded_cents is still 0, so two
        concurrent approvals cannot both succeed. Committed explicitly (pooled connections are
        rolled back on return otherwise)."""
        rid = _id("RFD")
        try:
            with self._conn.transaction():
                cur = self._conn.execute(
                    "UPDATE orders SET status='refunded', refunded_cents = refunded_cents + %s "
                    "WHERE order_id=%s AND tenant_id=%s AND refunded_cents = 0",
                    [amount_cents, order.order_id, order.tenant_id],
                )
                if cur.rowcount != 1:
                    raise RefundConflictError(order.order_id)
                self._conn.execute(
                    "INSERT INTO refunds (refund_id, tenant_id, order_id, amount_cents, reason, approved_by) "
                    "VALUES (%s,%s,%s,%s,%s,%s)",
                    [rid, order.tenant_id, order.order_id, amount_cents, reason, approved_by],
                )
            self._conn.commit()
        except RefundConflictError:
            self._conn.rollback()
            raise
        return rid
