"""Synthetic data: deterministic, sized as specified, and the demo order 1042 always exists."""

from __future__ import annotations

from datetime import date

from store_support.data.seed import generate


def test_sizes() -> None:
    d = generate(seed=1)
    assert len(d.customers) == 200
    assert len(d.orders) == 500
    assert len({p.sku for p in d.products}) == 50  # 50 SKUs, one row per tenant
    assert len(d.inventory) == len(d.products)


def test_deterministic() -> None:
    assert generate(seed=7).orders[:5] == generate(seed=7).orders[:5]


def test_two_tenants_and_every_row_has_tenant() -> None:
    d = generate(seed=1)
    tenants = {c.tenant_id for c in d.customers}
    assert tenants == {"store-a", "store-b"}
    assert all(o.tenant_id in tenants for o in d.orders)


def test_orders_reference_customers_in_same_tenant() -> None:
    d = generate(seed=1)
    by_id = {c.customer_id: c for c in d.customers}
    assert all(by_id[o.customer_id].tenant_id == o.tenant_id for o in d.orders)


def test_demo_order_1042_is_a_delivered_steering_wheel_cover_in_store_a() -> None:
    d = generate(seed=1)
    o = next(o for o in d.orders if o.order_id == 1042)
    assert o.tenant_id == "store-a"
    assert o.status == "delivered"
    assert "steering wheel cover" in d.product_name(o.tenant_id, o.sku).lower()
    assert 0 <= (date.today() - o.ordered_on).days <= 7  # inside the damaged-on-arrival window whenever the demo runs
