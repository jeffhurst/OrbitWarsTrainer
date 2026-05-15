"""Runnable training entrypoint for the first project version."""
from __future__ import annotations

from pathlib import Path

from .self_play import bootstrap_random_policy


def train(out: str | Path = "runs/models/bootstrap_policy.zip", seed: int = 0) -> Path:
    """Create a valid policy artifact.

    This keeps the pipeline runnable before full PPO integration. It produces a 15->9 policy file
    consumed by evaluation/watch/export scripts.
    """
    return bootstrap_random_policy(out, seed)
