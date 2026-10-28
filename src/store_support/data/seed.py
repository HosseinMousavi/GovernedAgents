"""Synthetic data with Faker: 2 tenants, 200 customers, 50 products, 500 orders, inventory.

`generate()` is pure (seeded) so tests and evals get identical data. `main()` loads it into
Postgres. Order 1042 is pinned to the demo scenario (damaged steering wheel cover, store-a).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

TENANTS = {"store-a": "AutoTrim Outfitters", "store-b": "RoadReady Parts"}

CATALOG: list[tuple[str, str, int, int]] = [  # (name, category, price_cents, warranty_months)
    ("Steering Wheel Cover - Leather", "interior", 2499, 12),
    ("Steering Wheel Cover - Microfiber", "interior", 1899, 12),
    ("All-Weather Floor Mats (set)", "interior", 7999, 24),
    ("Seat Cover Set - Neoprene", "interior", 8999, 12),
    ("Dash Cam 2K", "electronics", 12999, 24),
    ("OBD-II Bluetooth Scanner", "electronics", 3499, 12),
    ("Portable Tire Inflator", "tools", 4599, 24),
    ("Jump Starter 2000A", "tools", 8999, 24),
    ("Roof Rack Crossbars", "exterior", 15999, 36),
    ("Cargo Organizer", "interior", 2999, 12),
    ("Wiper Blades (pair)", "maintenance", 2299, 6),
    ("Cabin Air Filter", "maintenance", 1599, 6),
    ("Engine Air Filter", "maintenance", 1899, 6),
    ("Synthetic Motor Oil 5W-30 5qt", "maintenance", 3299, 0),
    ("Brake Pads - Ceramic Front", "parts", 5499, 24),
    ("Brake Rotors Front (pair)", "parts", 9999, 24),
    ("Car Battery Group 35", "parts", 15999, 36),
    ("Headlight Bulbs H11 LED (pair)", "lighting", 3999, 12),
    ("LED Light Bar 20in", "lighting", 8999, 24),
    ("Phone Mount - Vent", "accessories", 1499, 12),
    ("Wireless Charger Mount", "accessories", 3499, 12),
    ("Trunk Liner", "interior", 4999, 12),
    ("Sunshade Windshield", "accessories", 1999, 6),
    ("Car Cover - Outdoor", "exterior", 6999, 12),
    ("Mud Flaps (set)", "exterior", 3999, 24),
    ("Tow Hitch Ball Mount", "exterior", 5999, 36),
    ("Trailer Wiring Harness", "exterior", 4499, 12),
    ("Bike Rack - Hitch 2", "exterior", 19999, 36),
    ("Spark Plugs Iridium (4)", "parts", 3999, 12),
    ("Serpentine Belt", "parts", 2999, 12),
    ("Radiator Coolant 1gal", "maintenance", 2199, 0),
    ("Windshield Washer Fluid", "maintenance", 599, 0),
    ("Detailing Kit", "care", 4999, 0),
    ("Microfiber Towels (12)", "care", 1999, 0),
    ("Ceramic Spray Coating", "care", 2999, 0),
    ("Tire Shine", "care", 1299, 0),
    ("Car Vacuum 12V", "electronics", 3999, 12),
    ("Backup Camera Kit", "electronics", 7999, 24),
    ("Bluetooth FM Transmitter", "electronics", 2499, 12),
    ("Seat Cushion - Gel", "interior", 3499, 12),
    ("Steering Wheel Lock", "security", 4499, 24),
    ("Wheel Locks (set)", "security", 2999, 24),
    ("Emergency Kit", "safety", 5999, 12),
    ("First Aid Kit", "safety", 2499, 0),
    ("Reflective Triangles (3)", "safety", 1999, 0),
    ("Child Seat Protector", "interior", 1999, 12),
    ("Pet Seat Cover", "interior", 3499, 12),
    ("Tonneau Cover Soft", "exterior", 24999, 36),
    ("Running Boards (pair)", "exterior", 29999, 36),
    ("Fuel Injector Cleaner", "maintenance", 1099, 0),
]
assert len(CATALOG) == 50


@dataclass(frozen=True, slots=True)
class Customer:
    customer_id: str
    tenant_id: str
    full_name: str
    email: str
    phone: str


@dataclass(frozen=True, slots=True)
class Product:
    sku: str
    tenant_id: str
    name: str
    category: str
    price_cents: int
    warranty_months: int


@dataclass(frozen=True, slots=True)
class Inventory:
    tenant_id: str
    sku: str
    stock: int


@dataclass(frozen=True, slots=True)
class Order:
    order_id: int
    tenant_id: str
    customer_id: str
    sku: str
    total_cents: int
    status: str
    ordered_on: date


@dataclass(slots=True)
class Dataset:
    customers: list[Customer] = field(default_factory=list)
    products: list[Product] = field(default_factory=list)
    inventory: list[Inventory] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)

    def product_name(self, tenant_id: str, sku: str) -> str:
        return next(p.name for p in self.products if p.tenant_id == tenant_id and p.sku == sku)


def generate(*, seed: int = 42, n_customers: int = 200, n_orders: int = 500) -> Dataset:
    fake = Faker()
    Faker.seed(seed)
    rng = random.Random(seed)  # noqa: S311 - synthetic data, not security
    d = Dataset()

    tenant_ids = list(TENANTS)
    for i in range(n_customers):
        d.customers.append(Customer(
            customer_id=f"cust-{i + 1:04d}", tenant_id=tenant_ids[i % 2], full_name=fake.name(),
            email=fake.unique.email(), phone=fake.phone_number(),
        ))

    # Same 50-product catalog in each tenant; one Product row per tenant per SKU.
    for i, (name, cat, price, warr) in enumerate(CATALOG):
        sku = f"SKU-{i + 1:03d}"
        for t in tenant_ids:
            d.products.append(Product(sku, t, name, cat, price, warr))
            d.inventory.append(Inventory(t, sku, rng.randint(0, 40)))

    by_tenant = {t: [c for c in d.customers if c.tenant_id == t] for t in tenant_ids}
    statuses = ["placed", "shipped", "delivered", "delivered", "delivered", "cancelled"]
    today = date(2026, 10, 27)
    for n in range(n_orders):
        order_id = 1000 + n
        cust = rng.choice(d.customers)
        sku_idx = rng.randrange(len(CATALOG))
        price = CATALOG[sku_idx][2]
        d.orders.append(Order(
            order_id=order_id, tenant_id=cust.tenant_id, customer_id=cust.customer_id,
            sku=f"SKU-{sku_idx + 1:03d}", total_cents=price, status=rng.choice(statuses),
            ordered_on=today - timedelta(days=rng.randint(1, 400)),
        ))

    # Pin the demo order.
    demo_cust = by_tenant["store-a"][0]
    d.orders = [o for o in d.orders if o.order_id != 1042]
    d.orders.append(Order(1042, "store-a", demo_cust.customer_id, "SKU-001", 2499, "delivered", today - timedelta(days=6)))
    d.orders.sort(key=lambda o: o.order_id)
    return d


def load(dataset: Dataset, database_url: str, schema_path: Path) -> None:
    import psycopg

    with psycopg.connect(database_url) as conn:
        conn.execute(schema_path.read_text())
        for t, name in TENANTS.items():
            conn.execute("INSERT INTO tenants VALUES (%s,%s) ON CONFLICT DO NOTHING", [t, name])
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO customers (customer_id, tenant_id, full_name, email, phone) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                [(c.customer_id, c.tenant_id, c.full_name, c.email, c.phone) for c in dataset.customers],
            )
            cur.executemany(
                "INSERT INTO products (sku, tenant_id, name, category, price_cents, warranty_months) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                [(p.sku, p.tenant_id, p.name, p.category, p.price_cents, p.warranty_months) for p in dataset.products],
            )
            cur.executemany(
                "INSERT INTO inventory (tenant_id, sku, stock) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                [(i.tenant_id, i.sku, i.stock) for i in dataset.inventory],
            )
            cur.executemany(
                "INSERT INTO orders (order_id, tenant_id, customer_id, sku, total_cents, status, ordered_on) VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                [(o.order_id, o.tenant_id, o.customer_id, o.sku, o.total_cents, o.status, o.ordered_on) for o in dataset.orders],
            )
        conn.commit()


def main() -> None:
    from store_support.config import Settings

    schema = Path(__file__).resolve().parents[3] / "data" / "schema.sql"
    d = generate()
    load(d, Settings.from_env().database_url, schema)
    print(f"seeded {len(d.customers)} customers, {len(d.products)} product rows, {len(d.orders)} orders", file=sys.stderr)


if __name__ == "__main__":
    main()
