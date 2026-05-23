"""Build the 15-value per-source model observation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


from .actions import filter_candidates_with_valid_trajectories
from .candidates import CandidateConfig, comet_ids_from_obs, select_candidates
from .geometry import DEFAULT_SUN_COLLISION_RADIUS, LaunchSolution
from .planets import ownership_encoding, total_production
from .state_tracker import ProductionTracker
from .types import Planet, parse_planets
from .vector import FloatVector


@dataclass(slots=True)
class ObservationBuilder:
    candidate_config: CandidateConfig = CandidateConfig()
    tracker: ProductionTracker | None = None

    def build_for_source(
        self,
        source: Planet,
        planets: list[Planet],
        player: int,
        previous_total_production: float | None = None,
        comet_ids: set[int] | None = None,
        candidates: Sequence[Planet] | None = None,
    ) -> tuple[FloatVector, list[Planet]]:
        total = total_production(planets, player)
        if previous_total_production is None:
            delta = self.tracker.change(player, total) if self.tracker else 0.0
        else:
            delta = total - previous_total_production
        if candidates is None:
            selected = select_candidates(source, planets, player, comet_ids, self.candidate_config)
            chosen = selected.candidates
        else:
            chosen = list(candidates)[: self.candidate_config.max_candidates]
        values: list[float] = [float(total), float(delta), float(source.ships)]
        for idx in range(self.candidate_config.max_candidates):
            if idx < len(chosen):
                p = chosen[idx]
                enc = ownership_encoding(p, player)
                ships = float(p.ships if enc == 1 else -p.ships)
                values.extend([float(enc), ships, float(p.production)])
            else:
                values.extend([0.0, 0.0, 0.0])
        return FloatVector(float(v) for v in values), chosen

    def build_filtered_for_source(
        self,
        source: Planet,
        planets: list[Planet],
        player: int,
        previous_total_production: float | None = None,
        comet_ids: set[int] | None = None,
        *,
        angular_velocity: float = 0.0,
        fleet_speed: float = 1.0,
        sun_radius: float = DEFAULT_SUN_COLLISION_RADIUS,
    ) -> tuple[FloatVector, list[Planet], list[LaunchSolution]]:
        raw_candidates = select_candidates(
            source, planets, player, comet_ids, self.candidate_config
        ).candidates
        filtered_candidates, proposed_launches = filter_candidates_with_valid_trajectories(
            source,
            raw_candidates,
            angular_velocity=angular_velocity,
            fleet_speed=fleet_speed,
            sun_radius=sun_radius,
            max_candidates=self.candidate_config.max_candidates,
        )
        vec, chosen = self.build_for_source(
            source,
            planets,
            player,
            previous_total_production=previous_total_production,
            comet_ids=comet_ids,
            candidates=filtered_candidates,
        )
        return vec, chosen, proposed_launches


def build_observation_from_obs(
    obs: dict,
    source: Planet,
    previous_total_production: float | None = None,
    tracker: ProductionTracker | None = None,
) -> tuple[FloatVector, list[Planet]]:
    planets = parse_planets(obs)
    player = int(obs.get("player", 0))
    builder = ObservationBuilder(tracker=tracker)
    return builder.build_for_source(
        source,
        planets,
        player,
        previous_total_production,
        comet_ids_from_obs(obs),
    )
