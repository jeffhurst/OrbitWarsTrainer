"""Simple reward utilities for rollout/evaluation bookkeeping."""
from __future__ import annotations

from orbit_wars_rl.core.types import parse_fleets, parse_planets


def player_score(obs: dict, player: int) -> float:
    planets = parse_planets(obs)
    fleets = parse_fleets(obs)
    return float(sum(p.ships for p in planets if p.owner == player) + sum(f.ships for f in fleets if f.owner == player))


def production_controlled(obs: dict, player: int) -> float:
    return float(sum(p.production for p in parse_planets(obs) if p.owner == player))
