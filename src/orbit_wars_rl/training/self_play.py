"""Minimal self-play scaffolding.

This is intentionally simple for v1: the project first validates candidate selection visually, then
can swap this loop for a richer Kaggle/SB3 wrapper.
"""
from __future__ import annotations

from pathlib import Path

from orbit_wars_rl.models.policy import NumpyPolicy
from orbit_wars_rl.models.save_load import save_policy


def bootstrap_random_policy(out: str | Path = "runs/models/bootstrap_policy.zip", seed: int = 0) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_policy(NumpyPolicy.random(seed), path)
    return path
