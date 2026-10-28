"""agents/<name>/manifest.yaml: name, model, tools, memory, owner. Loaded and enforced."""

from __future__ import annotations

from pathlib import Path

import pytest

from store_support.governance.manifest import AgentManifest, ManifestError, load_all, load_manifest

AGENTS = Path(__file__).resolve().parents[2] / "agents"


def test_all_shipped_manifests_load() -> None:
    ms = load_all(AGENTS)
    assert {m.name for m in ms} == {"intake-agent", "knowledge-agent", "resolver-agent"}


def test_manifest_fields() -> None:
    m = load_manifest(AGENTS / "resolver-agent" / "manifest.yaml")
    assert m.model and m.owner and m.version
    assert "lookup_order" in m.tools
    assert m.memory.kind in {"none", "session", "persistent"}


def test_manifest_tools_must_exist_in_registry(tmp_path: Path) -> None:
    (tmp_path / "manifest.yaml").write_text(
        "name: x\nversion: 1.0.0\nmodel: m\nowner: o@team\ntools: [launch_missiles]\nmemory: {kind: none}\n"
    )
    with pytest.raises(ManifestError, match="launch_missiles"):
        load_manifest(tmp_path / "manifest.yaml")


def test_manifest_allows_tool_is_the_enforcement_point() -> None:
    m = AgentManifest(name="a", version="1", model="m", owner="o", tools=["lookup_order"], description="")
    assert m.allows("lookup_order") is True
    assert m.allows("issue_refund") is False


def test_knowledge_agent_has_no_business_tools() -> None:
    m = load_manifest(AGENTS / "knowledge-agent" / "manifest.yaml")
    assert m.tools == []
