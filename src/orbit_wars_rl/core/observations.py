"""Build the 15-value per-source model observation."""
from __future__ import annotations

from dataclasses import dataclass


from .candidates import CandidateConfig, comet_ids_from_obs, select_candidates
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
    ) -> tuple[FloatVector, list[Planet]]:
        total = total_production(planets, player)
        if previous_total_production is None:
            delta = self.tracker.change(player, total) if self.tracker else 0.0
        else:
            delta = total - previous_total_production
        chosen = select_candidates(source, planets, player, comet_ids, self.candidate_config).candidates
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


def build_observation_from_obs(
    obs: dict,
    source: Planet,
    previous_total_production: float | None = None,
    tracker: ProductionTracker | None = None,
) -> tuple[FloatVector, list[Planet]]:
    planets = parse_planets(obs)
    player = int(obs.get("player", 0))
    builder = ObservationBuilder(tracker=tracker)
    return builder.build_for_source(source, planets, player, previous_total_production, comet_ids_from_obs(obs))
