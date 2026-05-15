"""Kaggle-compatible fallback submission entrypoint.

The export script writes a standalone file. This module remains useful for local imports and as the
source of the fallback starter behavior.
"""
from __future__ import annotations

from typing import Any

from orbit_wars_rl.agents.starter_agent import StarterAgent

_AGENT = StarterAgent()


def agent(obs: dict[str, Any], config: Any | None = None) -> list[list[float | int]]:
    return _AGENT.act(obs)
