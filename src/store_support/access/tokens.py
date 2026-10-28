"""Scoped tokens: `base64url(json payload).hmac_sha256`.

Deliberately minimal (no JWT lib) so the whole trust chain fits on one screen.
Payload holds opaque ids and scopes only - never names, emails, or phone numbers.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Iterable

from store_support.access.identity import Principal
from store_support.access.rules import Scope


class TokenError(ValueError):
    """Raised for any invalid token. Message never echoes token contents."""


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(payload: str, key: str) -> str:
    return _b64(hmac.new(key.encode(), payload.encode(), hashlib.sha256).digest())


def issue_token(
    *,
    subject_id: str,
    tenant_id: str,
    scopes: Iterable[Scope | str],
    key: str,
    now: float | None = None,
    ttl_s: int = 3600,
) -> str:
    try:
        scope_values = sorted(Scope(s).value for s in scopes)
    except ValueError as e:
        raise TokenError("unknown scope") from e
    issued = int(now if now is not None else time.time())
    body = {"sub": subject_id, "tid": tenant_id, "scp": scope_values, "exp": issued + ttl_s}
    payload = _b64(json.dumps(body, separators=(",", ":"), sort_keys=True).encode())
    return f"{payload}.{_sign(payload, key)}"


def parse_token(token: str, *, key: str, now: float | None = None) -> Principal:
    try:
        payload, sig = token.rsplit(".", 1)
    except ValueError as e:
        raise TokenError("malformed token") from e
    if not hmac.compare_digest(sig, _sign(payload, key)):
        raise TokenError("bad signature")
    try:
        body = json.loads(_unb64(payload))
        exp = int(body["exp"])
        scopes = frozenset(Scope(s) for s in body["scp"])
        principal = Principal(subject_id=str(body["sub"]), tenant_id=str(body["tid"]), scopes=scopes)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
        raise TokenError("invalid payload") from e
    current = now if now is not None else time.time()
    if current > exp:
        raise TokenError("expired")
    return principal
