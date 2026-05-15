"""Placeholder wrappers for future SB3 or custom multi-agent integration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class OrbitWarsWrapper:
    env: Any

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        return self.env.reset(*args, **kwargs)

    def step(self, actions: Any) -> Any:
        return self.env.step(actions)
