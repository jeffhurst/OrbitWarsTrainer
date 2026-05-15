"""Planet collection helpers."""
from __future__ import annotations

from .types import NEUTRAL_OWNER, Planet


def owned_planets(planets: list[Planet], player: int) -> list[Planet]:
    return [p for p in planets if p.owner == player]


def total_production(planets: list[Planet], player: int) -> float:
    return float(sum(p.production for p in planets if p.owner == player))


def ownership_encoding(planet: Planet, player: int) -> int:
    if planet.owner == player:
        return 1
    if planet.owner == NEUTRAL_OWNER:
        return 0
    return -1


def owner_priority(planet: Planet, player: int) -> int:
    """Sort priority: enemy first, then neutral, then friendly."""
    enc = ownership_encoding(planet, player)
    return {-1: 0, 0: 1, 1: 2}[enc]
