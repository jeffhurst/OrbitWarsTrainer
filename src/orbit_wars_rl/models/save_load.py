"""Model save/load helpers."""
from __future__ import annotations

from pathlib import Path

from .policy import NumpyPolicy


def load_policy(path: str | Path | None) -> NumpyPolicy:
    if path is None:
        return NumpyPolicy.random(0)
    return NumpyPolicy.load(path)


def save_policy(policy: NumpyPolicy, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    policy.save(path)
