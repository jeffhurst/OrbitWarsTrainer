"""Per-owned-planet model agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from orbit_wars_rl.core.actions import ActionDecodeConfig, decode_model_outputs
from orbit_wars_rl.core.candidates import CandidateConfig, comet_ids_from_obs
from orbit_wars_rl.core.comets import CometController
from orbit_wars_rl.core.geometry import sun_collision_radius_from_obs
from orbit_wars_rl.core.observations import ObservationBuilder
from orbit_wars_rl.core.planets import total_production
from orbit_wars_rl.core.state_tracker import ProductionTracker
from orbit_wars_rl.core.types import Action, parse_planets, rows
from orbit_wars_rl.models.policy import NumpyPolicy


class Policy(Protocol):
    def predict(self, obs): ...


@dataclass(slots=True)
class ModelAgent:
    policy: Policy = field(default_factory=lambda: NumpyPolicy.random(0))
    candidate_config: CandidateConfig = CandidateConfig()
    action_config: ActionDecodeConfig = ActionDecodeConfig()
    tracker: ProductionTracker = field(default_factory=ProductionTracker)
    comet_controller: CometController = field(default_factory=CometController)

    def act(self, obs: dict[str, Any]) -> list[list[float | int]]:
        planets = parse_planets(obs)
        player = int(obs.get("player", 0))
        comet_ids = comet_ids_from_obs(obs)
        angular_velocity = float(obs.get("angular_velocity", 0.0))
        fleet_speed = float(obs.get("fleet_speed", 1.0))
        sun_radius = sun_collision_radius_from_obs(obs)
        builder = ObservationBuilder(self.candidate_config)
        total = total_production(planets, player)
        delta = self.tracker.change(player, total)
        previous_total = total - delta
        actions: list[Action] = self.comet_controller.update_and_forced_actions(
            obs, player, angular_velocity=angular_velocity, fleet_speed=fleet_speed
        )
        for source in [p for p in planets if p.owner == player and p.id not in comet_ids]:
            model_obs, chosen = builder.build_for_source(
                source,
                planets,
                player,
                previous_total_production=previous_total,
                comet_ids=comet_ids,
            )
            outputs = self.policy.predict(model_obs)
            actions.extend(
                decode_model_outputs(
                    source,
                    chosen,
                    outputs,
                    self.action_config,
                    angular_velocity=angular_velocity,
                    fleet_speed=fleet_speed,
                    sun_radius=sun_radius,
                )
            )
        return rows(actions)
