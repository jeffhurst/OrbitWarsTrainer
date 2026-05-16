"""Best-effort comet handling isolated from normal candidate selection."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .candidates import comet_ids_from_obs
from .geometry import angle_between, distance
from .types import Action, Planet, parse_planets

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CometMemory:
    owner_by_id: dict[int, int] = field(default_factory=dict)
    ships_by_id: dict[int, int] = field(default_factory=dict)
    pending_by_player: dict[int, set[int]] = field(default_factory=dict)


def closest_non_comet_planet(comet: Planet, planets: list[Planet], comet_ids: set[int]) -> Planet | None:
    candidates = [p for p in planets if p.id != comet.id and p.id not in comet_ids]
    if not candidates:
        return None
    return min(candidates, key=lambda p: (distance(comet, p), p.id))


class CometController:
    """Detect likely comet captures and force a next-turn comet launch.

    Kaggle may not expose explicit swept-fleet events, so a comet ship increase or owner change is
    treated as a likely capture signal. If that comet is currently owned by this player on a later
    call, it sends ships to the closest non-comet planet.
    """

    def __init__(self, reserve_ships: int = 1) -> None:
        self.memory = CometMemory()
        self.reserve_ships = reserve_ships

    def update_and_forced_actions(self, obs: dict[str, Any], player: int | None = None) -> list[Action]:
        planets = parse_planets(obs)
        player_id = int(obs.get("player", 0) if player is None else player)
        comet_ids = comet_ids_from_obs(obs)
        by_id = {p.id: p for p in planets}

        forced: list[Action] = []
        pending = set(self.memory.pending_by_player.get(player_id, set()))
        for cid in sorted(pending):
            comet = by_id.get(cid)
            if comet and comet.owner == player_id and comet.ships > self.reserve_ships:
                target = closest_non_comet_planet(comet, planets, comet_ids)
                if target:
                    ships = comet.ships - max(0, self.reserve_ships)
                    forced.append(Action(comet.id, angle_between(comet, target), ships))
                    LOGGER.info("forced comet launch comet=%s target=%s ships=%s", comet.id, target.id, ships)
            self.memory.pending_by_player[player_id].discard(cid)

        for cid in comet_ids:
            comet = by_id.get(cid)
            if comet is None:
                self.memory.owner_by_id.pop(cid, None)
                self.memory.ships_by_id.pop(cid, None)
                continue
            old_owner = self.memory.owner_by_id.get(cid)
            old_ships = self.memory.ships_by_id.get(cid)
            captured = old_owner is not None and (comet.owner != old_owner or (old_ships is not None and comet.ships > old_ships + max(1, comet.production)))
            if captured and comet.owner >= 0:
                self.memory.pending_by_player.setdefault(comet.owner, set()).add(cid)
            self.memory.owner_by_id[cid] = comet.owner
            self.memory.ships_by_id[cid] = comet.ships
        return forced
