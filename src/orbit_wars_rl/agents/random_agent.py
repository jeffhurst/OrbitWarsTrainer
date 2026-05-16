"""Random baseline that samples selected candidate targets."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from orbit_wars_rl.core.candidates import CandidateConfig, comet_ids_from_obs, select_candidates
from orbit_wars_rl.core.geometry import (
    predict_launch,
    sun_collision_radius_from_obs,
    trajectory_crosses_sun,
)
from orbit_wars_rl.core.types import Action, parse_planets, rows


@dataclass(slots=True)
class RandomAgent:
    seed: int = 0
    launch_probability: float = 0.35
    max_fraction: float = 0.25
    candidate_config: CandidateConfig = CandidateConfig()
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def act(self, obs: dict[str, Any]) -> list[list[float | int]]:
        planets = parse_planets(obs)
        player = int(obs.get("player", 0))
        comet_ids = comet_ids_from_obs(obs)
        angular_velocity = float(obs.get("angular_velocity", 0.0))
        fleet_speed = float(obs.get("fleet_speed", 1.0))
        sun_radius = sun_collision_radius_from_obs(obs)
        actions: list[Action] = []
        for source in [p for p in planets if p.owner == player and p.id not in comet_ids]:
            if source.ships <= 1 or self.rng.random() > self.launch_probability:
                continue
            chosen = select_candidates(
                source, planets, player, comet_ids, self.candidate_config
            ).candidates
            if not chosen:
                continue
            target = self.rng.choice(chosen)
            launch = predict_launch(source, target, angular_velocity, fleet_speed)
            if trajectory_crosses_sun(launch.source_xy, launch.target_xy, sun_radius=sun_radius):
                continue
            ships = max(
                1, min(source.ships - 1, int(source.ships * self.rng.random() * self.max_fraction))
            )
            actions.append(Action(source.id, launch.angle, ships))
        return rows(actions)
