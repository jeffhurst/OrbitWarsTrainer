"""Simple heuristic baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orbit_wars_rl.core.candidates import CandidateConfig, comet_ids_from_obs, select_candidates
from orbit_wars_rl.core.geometry import launch_angle
from orbit_wars_rl.core.types import Action, parse_planets, rows


@dataclass(slots=True)
class HeuristicAgent:
    candidate_config: CandidateConfig = CandidateConfig()
    reserve_ships: int = 2

    def act(self, obs: dict[str, Any]) -> list[list[float | int]]:
        planets = parse_planets(obs)
        player = int(obs.get("player", 0))
        comet_ids = comet_ids_from_obs(obs)
        angular_velocity = float(obs.get("angular_velocity", 0.0))
        fleet_speed = float(obs.get("fleet_speed", 1.0))
        actions: list[Action] = []
        for source in [p for p in planets if p.owner == player and p.id not in comet_ids]:
            chosen = select_candidates(
                source, planets, player, comet_ids, self.candidate_config
            ).candidates
            if not chosen or source.ships <= self.reserve_ships:
                continue
            target = chosen[0]
            if target.owner == player:
                continue
            ships = min(source.ships - self.reserve_ships, max(1, abs(target.ships) + 1))
            actions.append(
                Action(
                    source.id, launch_angle(source, target, angular_velocity, fleet_speed), ships
                )
            )
        return rows(actions)
