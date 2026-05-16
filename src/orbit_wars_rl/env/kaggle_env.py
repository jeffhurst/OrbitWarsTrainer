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


def require_kaggle_env(**kwargs: Any):
    """Create the Kaggle Orbit Wars env or raise an actionable error."""
    try:
        from kaggle_environments import make  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Watching real Orbit Wars games requires kaggle-environments. "
            "Install it with `pip install -e .[kaggle]` or `pip install kaggle-environments`."
        ) from exc

    try:
        return make("orbit_wars", **kwargs)
    except Exception as exc:
        raise RuntimeError(
            "kaggle-environments is installed, but it could not create the `orbit_wars` "
            "environment. Upgrade kaggle-environments to a version that includes Orbit Wars."
        ) from exc
