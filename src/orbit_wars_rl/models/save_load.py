"""Model save/load helpers."""
from __future__ import annotations

from pathlib import Path
import zipfile

from .policy import NumpyPolicy
from .sb3_policy import SB3PolicyAdapter


def load_policy(path: str | Path | None):
    return load_any_policy(path)


def load_any_policy(path: str | Path | None):
    if path is None:
        return NumpyPolicy.random(0)
    if str(path).endswith(".zip") and zipfile.is_zipfile(path):
        return SB3PolicyAdapter.load(path)
    return NumpyPolicy.load(path)


def save_policy(policy: NumpyPolicy, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    policy.save(path)
