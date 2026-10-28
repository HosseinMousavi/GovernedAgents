"""Two identities, never merged.

Principal      = the *user* on whose behalf a request is made (carries scopes).
AgentIdentity  = the *software agent* executing the request (carries no scopes).

Both are logged, in separate fields, so an audit answers "which agent did what for whom".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from store_support.access.rules import Scope


@dataclass(frozen=True, slots=True)
class Principal:
    subject_id: str
    tenant_id: str
    scopes: frozenset[Scope] = field(default_factory=frozenset)

    def has(self, scope: Scope) -> bool:
        return scope in self.scopes


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    name: str
    version: str


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Everything the MCP server needs to authorize and audit one tool call."""

    principal: Principal
    agent: AgentIdentity
    request_id: str
