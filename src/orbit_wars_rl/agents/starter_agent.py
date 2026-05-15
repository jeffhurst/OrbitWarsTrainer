"""Starter visual-debug opponent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from orbit_wars_rl.core.candidates import CandidateConfig, comet_ids_from_obs, select_candidates
from orbit_wars_rl.core.comets import CometController
from orbit_wars_rl.core.geometry import angle_between
from orbit_wars_rl.core.types import Action, parse_planets, rows


@dataclass(slots=True)
class StarterAgent:
    candidate_config: CandidateConfig = CandidateConfig()
    ships_per_target: int = 1
    include_comet_forced_actions: bool = True
    comet_controller: CometController = field(default_factory=CometController)

    def act(self, obs: dict[str, Any]) -> list[list[float | int]]:
        planets = parse_planets(obs)
        player = int(obs.get("player", 0))
        comet_ids = comet_ids_from_obs(obs)
        actions: list[Action] = []
        if self.include_comet_forced_actions:
            actions.extend(self.comet_controller.update_and_forced_actions(obs, player))
        for source in [p for p in planets if p.owner == player]:
            # Comet launch behavior is handled separately.
            if source.id in comet_ids:
                continue
            chosen = select_candidates(source, planets, player, comet_ids, self.candidate_config).candidates
            available = source.ships
            if available < len(chosen) * self.ships_per_target:
                continue
            for target in chosen:
                actions.append(Action(source.id, angle_between(source, target), self.ships_per_target))
        return rows(actions)


def agent(obs: dict[str, Any], config: Any | None = None) -> list[list[float | int]]:
    return StarterAgent().act(obs)
