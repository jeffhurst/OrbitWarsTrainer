"""Evaluation metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class EvaluationMetrics:
    games: int = 0
    win_rate: float = 0.0
    average_final_score: float = 0.0
    average_production_controlled: float = 0.0
    average_ships_controlled: float = 0.0
    average_planets_controlled: float = 0.0
    average_number_of_launches: float = 0.0
    average_invalid_actions: float = 0.0
    average_episode_length: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)
