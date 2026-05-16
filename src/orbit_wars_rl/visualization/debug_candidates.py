"""Candidate-selection debug helpers."""
from __future__ import annotations

from pathlib import Path

from .demo import make_demo_observation
from .render import render_observation


def render_debug_candidates(out: str | Path = "runs/watch/debug_candidates.png") -> Path:
    return render_observation(make_demo_observation(0), out, title="Candidate selection debug")
