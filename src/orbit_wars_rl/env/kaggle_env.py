"""Optional Kaggle environment adapter."""
from __future__ import annotations

from typing import Any


def make_kaggle_env(**kwargs: Any):
    """Create Kaggle Orbit Wars env if installed; otherwise return None."""
    try:
        from kaggle_environments import make  # type: ignore
    except Exception:
        return None
    try:
        return make("orbit_wars", **kwargs)
    except Exception:
        return None
