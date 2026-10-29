"""Settings from environment. One object, read once, passed explicitly (no global state in logic)."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEV_KEY = "dev-only-change-me"


class SettingsError(RuntimeError):
    pass


def _bool(v: str | None, default: bool = False) -> bool:
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    token_signing_key: str
    kill_switch: bool
    anthropic_api_key: str
    openai_api_key: str
    ollama_base_url: str
    embedding_model: str
    mcp_server_url: str
    otel_endpoint: str
    otel_service_name: str
    n8n_webhook_url: str
    chat_model: str
    routing_policy: str  # routed | all_default | all_cheap
    cheap_model: str  # optional override of the manifest's cheap tier (e.g. ollama/ssa-grader)
    budget_soft_usd: float  # per tenant per day: degrade to cheap tier
    budget_hard_usd: float  # per tenant per day: refuse LLM work (0 = no budget)

    @classmethod
    def from_env(cls, *, strict: bool = False) -> Settings:
        """strict=True (used by the long-running services) refuses to start with a missing or
        default signing key - otherwise anyone could mint manager:refund tokens."""
        e = os.environ.get
        key = e("TOKEN_SIGNING_KEY", DEV_KEY)
        if strict and (key == DEV_KEY or len(key) < 16):
            raise SettingsError("TOKEN_SIGNING_KEY must be set to a secret of at least 16 characters")
        return cls(
            database_url=e("DATABASE_URL", "postgresql://ssa:ssa@localhost:5432/ssa"),
            token_signing_key=key,
            kill_switch=_bool(e("KILL_SWITCH")),
            anthropic_api_key=e("ANTHROPIC_API_KEY", ""),
            openai_api_key=e("OPENAI_API_KEY", ""),
            ollama_base_url=e("OLLAMA_BASE_URL", "http://localhost:11434"),
            embedding_model=e("EMBEDDING_MODEL", "bge-m3"),
            mcp_server_url=e("MCP_SERVER_URL", "http://localhost:8765/mcp"),
            otel_endpoint=e("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            otel_service_name=e("OTEL_SERVICE_NAME", "store-support-agent"),
            n8n_webhook_url=e("N8N_WEBHOOK_URL", "http://localhost:5678/webhook"),
            chat_model=e("CHAT_MODEL", "claude-sonnet-4-5"),
            routing_policy=e("ROUTING_POLICY", "routed"),
            cheap_model=e("CHEAP_MODEL", ""),
            budget_soft_usd=float(e("BUDGET_SOFT_USD_PER_DAY", "5")),
            budget_hard_usd=float(e("BUDGET_HARD_USD_PER_DAY", "10")),
        )
