"""KILL_SWITCH: when engaged, no tool executes anywhere (MCP server AND agent-side).

Two sources, either engages it:
  * env KILL_SWITCH=true   (set at deploy time; ECS task env / docker-compose)
  * a flag file            (flip at runtime without a restart: `touch /tmp/ssa.KILL`)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

DEFAULT_FLAG = Path(os.environ.get("KILL_SWITCH_FILE", "/tmp/ssa.KILL"))  # noqa: S108


class KillSwitch:
    def __init__(self, env: Mapping[str, str] | None = None, flag_file: Path | None = None) -> None:
        self._env = os.environ if env is None else env
        self._flag = DEFAULT_FLAG if flag_file is None else flag_file

    @property
    def engaged(self) -> bool:
        if self._env.get("KILL_SWITCH", "").strip().lower() in {"1", "true", "yes", "on"}:
            return True
        return self._flag.exists()
