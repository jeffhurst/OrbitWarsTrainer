"""Typed wrappers for Orbit Wars observations and actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

NEUTRAL_OWNER = -1


@dataclass(frozen=True, slots=True)
class Planet:
    """Planet row: [id, owner, x, y, radius, ships, production]."""

    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: float

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> "Planet":
        return cls(
            id=int(row[0]),
            owner=int(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            radius=float(row[4]),
            ships=int(row[5]),
            production=float(row[6]),
        )

    def to_row(self) -> list[float | int]:
        return [self.id, self.owner, self.x, self.y, self.radius, self.ships, self.production]


@dataclass(frozen=True, slots=True)
class Fleet:
    """Fleet row: [id, owner, x, y, angle, from_planet_id, ships]."""

    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> "Fleet":
        return cls(int(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4]), int(row[5]), int(row[6]))


@dataclass(frozen=True, slots=True)
class Action:
    """Fleet launch action in Kaggle format."""

    from_planet_id: int
    direction_angle: float
    num_ships: int

    def to_row(self) -> list[float | int]:
        return [self.from_planet_id, self.direction_angle, self.num_ships]


def parse_planets(obs: dict[str, Any] | Any) -> list[Planet]:
    data = obs.get("planets", []) if hasattr(obs, "get") else getattr(obs, "planets", [])
    return [p if isinstance(p, Planet) else Planet.from_row(p) for p in data]


def parse_fleets(obs: dict[str, Any] | Any) -> list[Fleet]:
    data = obs.get("fleets", []) if hasattr(obs, "get") else getattr(obs, "fleets", [])
    return [f if isinstance(f, Fleet) else Fleet.from_row(f) for f in data]


def current_player(obs: dict[str, Any] | Any, default: int = 0) -> int:
    return int(obs.get("player", default) if hasattr(obs, "get") else getattr(obs, "player", default))


def rows(actions: Iterable[Action]) -> list[list[float | int]]:
    return [a.to_row() if isinstance(a, Action) else list(a) for a in actions]  # type: ignore[arg-type]
