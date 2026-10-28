"""Access log. One entry per tool call, agent and user in separate fields, subject id hashed.

The entries are plain dicts so they can be shipped to structlog/CloudWatch unchanged.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import structlog

from store_support.access import DenyReason, RequestContext
from store_support.governance.redaction import redact_text

log = structlog.get_logger("access")


def pseudonymize(subject_id: str, *, salt: str = "ssa") -> str:
    """Stable, non-reversible handle so audit rows join across requests without storing the raw id."""
    return "u_" + hashlib.sha256(f"{salt}:{subject_id}".encode()).hexdigest()[:12]


class AuditLog:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def record(
        self,
        ctx: RequestContext,
        *,
        tool: str,
        outcome: str,
        reason: DenyReason | str | None = None,
        detail: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "ts": time.time(),
            "request_id": ctx.request_id,
            "agent": f"{ctx.agent.name}@{ctx.agent.version}",
            "subject": pseudonymize(ctx.principal.subject_id),
            "tenant": ctx.principal.tenant_id,
            "scopes": sorted(s.value for s in ctx.principal.scopes),
            "tool": tool,
            "outcome": outcome,
            "reason": str(reason) if reason else None,
        }
        if detail:
            entry["detail"] = redact_text(detail)
        self.entries.append(entry)
        log.info("tool_call", **entry)
