"""Simple heuristic baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orbit_wars_rl.core.candidates import CandidateConfig, comet_ids_from_obs, select_candidates
from orbit_wars_rl.core.geometry import (
    predict_launch,
    sun_collision_radius_from_obs,
    trajectory_crosses_sun,
)
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
        sun_radius = sun_collision_radius_from_obs(obs)
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
            launch = predict_launch(source, target, angular_velocity, fleet_speed)
            if trajectory_crosses_sun(launch.source_xy, launch.target_xy, sun_radius=sun_radius):
                continue
            ships = min(source.ships - self.reserve_ships, max(1, abs(target.ships) + 1))
            actions.append(Action(source.id, launch.angle, ships))
        return rows(actions)


@dataclass(slots=True)
class GreedyAgent:
    """Greedy baseline: attacks the highest-priority non-owned candidate with most available ships."""

    candidate_config: CandidateConfig = CandidateConfig()
    reserve_ships: int = 1

    def act(self, obs: dict[str, Any]) -> list[list[float | int]]:
        planets = parse_planets(obs)
        player = int(obs.get("player", 0))
        comet_ids = comet_ids_from_obs(obs)
        angular_velocity = float(obs.get("angular_velocity", 0.0))
        fleet_speed = float(obs.get("fleet_speed", 1.0))
        sun_radius = sun_collision_radius_from_obs(obs)
        actions: list[Action] = []
        for source in [p for p in planets if p.owner == player and p.id not in comet_ids]:
            if source.ships <= self.reserve_ships:
                continue
            chosen = select_candidates(source, planets, player, comet_ids, self.candidate_config).candidates
            target = next((c for c in chosen if c.owner != player), None)
            if target is None:
                continue
            launch = predict_launch(source, target, angular_velocity, fleet_speed)
            if trajectory_crosses_sun(launch.source_xy, launch.target_xy, sun_radius=sun_radius):
                continue
            ships = max(1, source.ships - self.reserve_ships)
            actions.append(Action(source.id, launch.angle, ships))
        return rows(actions)


@dataclass(slots=True)
class HardAgent:
    """Stronger scripted opponent that can issue multiple attacks per turn when possible."""

    candidate_config: CandidateConfig = CandidateConfig()
    reserve_ships: int = 2
    max_attacks_per_source: int = 2

    def act(self, obs: dict[str, Any]) -> list[list[float | int]]:
        planets = parse_planets(obs)
        player = int(obs.get("player", 0))
        comet_ids = comet_ids_from_obs(obs)
        angular_velocity = float(obs.get("angular_velocity", 0.0))
        fleet_speed = float(obs.get("fleet_speed", 1.0))
        sun_radius = sun_collision_radius_from_obs(obs)
        actions: list[Action] = []
        for source in [p for p in planets if p.owner == player and p.id not in comet_ids]:
            available = source.ships - self.reserve_ships
            if available <= 0:
                continue
            chosen = select_candidates(source, planets, player, comet_ids, self.candidate_config).candidates
            attack_targets = [c for c in chosen if c.owner != player][: self.max_attacks_per_source]
            for target in attack_targets:
                if available <= 0:
                    break
                launch = predict_launch(source, target, angular_velocity, fleet_speed)
                if trajectory_crosses_sun(launch.source_xy, launch.target_xy, sun_radius=sun_radius):
                    continue
                needed = max(1, abs(target.ships) + 2)
                ships = min(available, needed)
                actions.append(Action(source.id, launch.angle, ships))
                available -= ships
        return rows(actions)
