"""Training callback placeholders kept separate from core logic."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainingLog:
    step: int
    reward: float
