"""Geometry helpers for Orbit Wars' screen-coordinate board.

The board origin is top-left: x increases right and y increases downward. Orbit Wars launch
angles use 0 for right and pi/2 for down, so ``math.atan2(target.y-source.y, target.x-source.x)``
produces the correct action angle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .types import Planet

BOARD_SIZE = 100.0
CENTER = (50.0, 50.0)
ROTATION_RADIUS_LIMIT = 50.0


@dataclass(frozen=True, slots=True)
class QuadrantConfig:
    center: tuple[float, float] = CENTER
    ccw_map: dict[int, int] | None = None

    def next_ccw(self, quadrant: int) -> int:
        mapping = self.ccw_map or {1: 2, 2: 3, 3: 4, 4: 1}
        return mapping[quadrant]


def distance_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def distance(a: Planet, b: Planet) -> float:
    return distance_xy(a.x, a.y, b.x, b.y)


def distance_from_center(p: Planet, center: tuple[float, float] = CENTER) -> float:
    return distance_xy(p.x, p.y, center[0], center[1])


def angle_between(source: Planet, target: Planet) -> float:
    """Return launch angle where 0=right and pi/2=down."""
    return math.atan2(target.y - source.y, target.x - source.x)


def is_orbiting_planet(
    planet: Planet,
    center: tuple[float, float] = CENTER,
    rotation_radius_limit: float = ROTATION_RADIUS_LIMIT,
) -> bool:
    """Likely Orbit Wars orbiting rule, isolated for easy replacement.

    The public game description says orbiting planets satisfy
    ``distance_from_center + planet.radius < 50``.
    """
    return distance_from_center(planet, center) + planet.radius < rotation_radius_limit


def quadrant_of(x: float, y: float, config: QuadrantConfig | None = None) -> int:
    """Return screen-coordinate quadrant around center.

    Boundaries are intentional and tested:
    Q1: x >= cx and y < cy (top-right)
    Q2: x <  cx and y < cy (top-left)
    Q3: x <  cx and y >= cy (bottom-left)
    Q4: x >= cx and y >= cy (bottom-right)
    """
    cfg = config or QuadrantConfig()
    cx, cy = cfg.center
    if x >= cx and y < cy:
        return 1
    if x < cx and y < cy:
        return 2
    if x < cx and y >= cy:
        return 3
    return 4


def counterclockwise_quadrant(q: int, config: QuadrantConfig | None = None) -> int:
    return (config or QuadrantConfig()).next_ccw(q)
