"""Candidate target selection for per-source decisions."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .geometry import (
    CENTER,
    QuadrantConfig,
    counterclockwise_quadrant,
    distance,
    is_orbiting_planet,
    quadrant_of,
)
from .planets import owner_priority
from .types import Planet

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    static_radius: float = 25.0
    max_candidates: int = 4
    center: tuple[float, float] = CENTER
    rotation_radius_limit: float = 50.0
    quadrant_config: QuadrantConfig = QuadrantConfig()


@dataclass(frozen=True, slots=True)
class CandidateSet:
    candidates: list[Planet]
    static_candidates: list[Planet]
    orbiting_candidates: list[Planet]


def comet_ids_from_obs(obs: dict[str, Any] | None) -> set[int]:
    if not obs:
        return set()
    ids: set[int] = {int(pid) for pid in obs.get("comet_planet_ids", []) or []}
    for group in obs.get("comets", []) or []:
        if isinstance(group, dict):
            ids.update(int(pid) for pid in group.get("planet_ids", []) or [])
        elif isinstance(group, (list, tuple)) and group:
            # Best effort for tuple/list groups whose first field is planet ids.
            first = group[0]
            if isinstance(first, (list, tuple, set)):
                ids.update(int(pid) for pid in first)
    return ids


def is_comet(planet: Planet, comet_ids: set[int]) -> bool:
    return planet.id in comet_ids


def select_candidates(
    source: Planet,
    planets: list[Planet],
    player: int,
    comet_ids: set[int] | None = None,
    config: CandidateConfig | None = None,
) -> CandidateSet:
    """Select up to four targets for ``source``.

    Static planets are included if they are within ``static_radius``. Orbiting planets are
    included if they are in the counterclockwise screen-coordinate quadrant from the source.
    Comets and the source planet are always excluded.
    """
    cfg = config or CandidateConfig()
    comets = comet_ids or set()
    src_q = quadrant_of(source.x, source.y, cfg.quadrant_config)
    wanted_q = counterclockwise_quadrant(src_q, cfg.quadrant_config)
    source_orbiting = is_orbiting_planet(source, cfg.center, cfg.rotation_radius_limit)
    static: list[Planet] = []
    orbiting: list[Planet] = []
    seen: set[int] = set()

    for p in planets:
        if p.id == source.id or p.id in seen or is_comet(p, comets):
            continue
        orbiting_p = is_orbiting_planet(p, cfg.center, cfg.rotation_radius_limit)
        if not orbiting_p:
            if source_orbiting:
                if quadrant_of(p.x, p.y, cfg.quadrant_config) == src_q:
                    static.append(p)
                    seen.add(p.id)
            elif distance(source, p) <= cfg.static_radius:
                static.append(p)
                seen.add(p.id)
        elif orbiting_p:
            orbit_q = quadrant_of(p.x, p.y, cfg.quadrant_config)
            if orbit_q == wanted_q or (source_orbiting and orbit_q == src_q):
                orbiting.append(p)
            seen.add(p.id)

    combined = static + orbiting
    candidate_group = {p.id: 0 for p in static}
    candidate_group.update({p.id: 1 for p in orbiting})
    combined.sort(
        key=lambda p: (
            p.ships,
            0 if p.owner != player else 1,
            owner_priority(p, player),
            candidate_group[p.id],
            -p.production,
            distance(source, p),
            p.id,
        )
    )
    selected = combined[: cfg.max_candidates]
    LOGGER.debug(
        "source=%s static=%s orbiting=%s selected=%s",
        source.id,
        [p.id for p in static],
        [p.id for p in orbiting],
        [p.id for p in selected],
    )
    return CandidateSet(selected, static, orbiting)
