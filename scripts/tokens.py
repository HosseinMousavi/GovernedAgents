"""Print demo tokens for the three roles. `make tokens` -> paste into .env for n8n."""

import os

from store_support.access import Scope, issue_token

key = os.environ.get("TOKEN_SIGNING_KEY", "dev-only-change-me")
print(
    "CUSTOMER_TOKEN="
    + issue_token(
        subject_id="cust-0001", tenant_id="store-a", scopes=[Scope.CUSTOMER_READ_OWN], key=key, ttl_s=86400 * 30
    )
)
print(
    "STAFF_TOKEN="
    + issue_token(
        subject_id="staff-0001", tenant_id="store-a", scopes=[Scope.STAFF_READ_ALL], key=key, ttl_s=86400 * 30
    )
)
print(
    "MANAGER_TOKEN="
    + issue_token(
        subject_id="mgr-0001",
        tenant_id="store-a",
        scopes=[Scope.STAFF_READ_ALL, Scope.MANAGER_REFUND],
        key=key,
        ttl_s=86400 * 30,
    )
)
