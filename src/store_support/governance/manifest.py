"""Agent manifests: the registry of who runs, with which model, allowed to call which tools.

`AgentManifest.allows(tool)` is the enforcement point used by the resolver before any MCP call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from store_support.access import TOOL_SCOPES


class ManifestError(ValueError):
    pass


class Memory(BaseModel):
    kind: Literal["none", "session", "persistent"] = "none"
    ttl_minutes: int | None = None


class AgentManifest(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9-]+$")
    version: str
    description: str = ""
    owner: str
    model: str
    models: dict[str, str] = Field(default_factory=dict)  # optional {cheap, strong} tiers; `model` is the default
    runtime: str = "python"
    tools: list[str] = Field(default_factory=list)
    memory: Memory = Field(default_factory=Memory)
    limits: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tools")
    @classmethod
    def _tools_exist(cls, tools: list[str]) -> list[str]:
        unknown = [t for t in tools if t not in TOOL_SCOPES]
        if unknown:
            raise ValueError(f"unknown tools: {', '.join(unknown)}")
        return tools

    def allows(self, tool: str) -> bool:
        return tool in self.tools


def load_manifest(path: Path) -> AgentManifest:
    try:
        return AgentManifest.model_validate(yaml.safe_load(path.read_text()) or {})
    except (ValidationError, yaml.YAMLError) as e:
        raise ManifestError(f"{path}: {e}") from e


def load_all(root: Path) -> list[AgentManifest]:
    return [load_manifest(p) for p in sorted(root.glob("*/manifest.yaml"))]
