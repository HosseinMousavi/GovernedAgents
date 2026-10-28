"""Postgres repository against a real database. Skipped unless DATABASE_URL points at one
(CI's `services: postgres` job sets it; locally `make up` does)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.db

URL = os.environ.get("DATABASE_URL_TEST") or os.environ.get("DATABASE_URL")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def pool():
    if not URL:
        pytest.skip("no DATABASE_URL")
    from psycopg_pool import ConnectionPool

    from store_support.data.seed import generate, load

    try:
        load(generate(seed=1), URL, ROOT / "data" / "schema.sql")
    except psycopg.OperationalError as e:  # pragma: no cover
        pytest.skip(f"postgres unreachable: {e}")
    p = ConnectionPool(URL, min_size=1, max_size=3, open=True)
    yield p
    p.close()


def _repo(conn):
    from store_support.mcp_server.repo_postgres import PostgresRepository

    return PostgresRepository(conn)


def test_refund_is_committed_and_visible_from_another_connection(pool) -> None:
    from store_support.access import RowFilter

    with pool.connection() as c1:
        r = _repo(c1)
        o = r.get_order(1042, RowFilter("store-a", None))
        if o.refunded_cents:
            pytest.skip("demo order already refunded in this database")
        r.issue_refund(order=o, amount_cents=100, reason="t", approved_by="m")
    with pool.connection() as c2:
        o2 = _repo(c2).get_order(1042, RowFilter("store-a", None))
    assert o2.refunded_cents == 100 and o2.status == "refunded"


def test_second_refund_with_stale_snapshot_conflicts(pool) -> None:
    from store_support.access import RowFilter
    from store_support.mcp_server.repo import RefundConflictError

    with pool.connection() as c:
        r = _repo(c)
        o = r.get_order(1043, RowFilter("store-a", None)) or r.get_order(1043, RowFilter("store-b", None))
        assert o is not None
        from dataclasses import replace

        fresh = replace(o, refunded_cents=0)
        try:
            r.issue_refund(order=fresh, amount_cents=1, reason="a", approved_by="m")
        except RefundConflictError:
            pass  # already refunded by an earlier run - still the behaviour we want
        with pytest.raises(RefundConflictError):
            r.issue_refund(order=fresh, amount_cents=1, reason="b", approved_by="m")
