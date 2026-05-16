"""Deterministic observations for non-Kaggle debug helpers."""
from __future__ import annotations

from typing import Any


def make_demo_observation(player: int = 0) -> dict[str, Any]:
    """Return a tiny static board used by candidate/evaluation smoke tests."""
    return {
        "player": player,
        "angular_velocity": 0.03,
        "comet_planet_ids": [8],
        "comets": [{"planet_ids": [8], "paths": [], "path_index": 0}],
        "fleets": [],
        "planets": [
            [0, 0, 80, 20, 2.0, 10, 3],
            [1, 1, 20, 80, 2.0, 10, 3],
            [2, -1, 70, 18, 1.7, 8, 5],
            [3, -1, 62, 16, 1.5, 6, 4],
            [4, -1, 35, 35, 1.2, 5, 2],
            [5, -1, 20, 20, 1.2, 5, 5],
            [6, -1, 15, 82, 1.5, 8, 4],
            [7, -1, 75, 78, 1.5, 8, 4],
            [8, -1, 50, 8, 1.0, 4, 1],
        ],
        "initial_planets": [],
    }
