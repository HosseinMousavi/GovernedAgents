"""Scoped tokens: HMAC-signed, short-lived, carry subject/tenant/scopes only."""

from __future__ import annotations

import pytest

from store_support.access.identity import Principal
from store_support.access.rules import Scope
from store_support.access.tokens import TokenError, issue_token, parse_token

KEY = "unit-test-key"


def test_round_trip() -> None:
    tok = issue_token(subject_id="cust-1", tenant_id="store-a", scopes=[Scope.CUSTOMER_READ_OWN], key=KEY)
    p = parse_token(tok, key=KEY, now=1000)
    assert p == Principal(subject_id="cust-1", tenant_id="store-a", scopes=frozenset({Scope.CUSTOMER_READ_OWN}))


def test_tampered_signature_rejected() -> None:
    tok = issue_token(subject_id="cust-1", tenant_id="store-a", scopes=[Scope.CUSTOMER_READ_OWN], key=KEY)
    with pytest.raises(TokenError):
        parse_token(tok[:-2] + "zz", key=KEY)


def test_tampered_payload_rejected() -> None:
    tok = issue_token(subject_id="cust-1", tenant_id="store-a", scopes=[Scope.CUSTOMER_READ_OWN], key=KEY)
    payload, sig = tok.rsplit(".", 1)
    with pytest.raises(TokenError):
        parse_token(payload + "x." + sig, key=KEY)


def test_wrong_key_rejected() -> None:
    tok = issue_token(subject_id="cust-1", tenant_id="store-a", scopes=[Scope.CUSTOMER_READ_OWN], key=KEY)
    with pytest.raises(TokenError):
        parse_token(tok, key="other")


def test_expired_rejected() -> None:
    tok = issue_token(
        subject_id="cust-1", tenant_id="store-a", scopes=[Scope.CUSTOMER_READ_OWN], key=KEY, now=0, ttl_s=60
    )
    with pytest.raises(TokenError):
        parse_token(tok, key=KEY, now=61)


def test_unknown_scope_string_rejected() -> None:
    with pytest.raises(TokenError):
        issue_token(subject_id="c", tenant_id="t", scopes=["root:everything"], key=KEY)  # type: ignore[list-item]


def test_token_does_not_embed_email_or_name() -> None:
    """Tokens carry opaque ids only; anything PII-shaped is a design bug."""
    tok = issue_token(subject_id="cust-1", tenant_id="store-a", scopes=[Scope.CUSTOMER_READ_OWN], key=KEY)
    assert "@" not in tok
