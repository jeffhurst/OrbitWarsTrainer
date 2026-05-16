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
SUN_RADIUS = 8.0


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


def future_orbit_position(
    planet: Planet,
    dt: float,
    angular_velocity: float,
    center: tuple[float, float] = CENTER,
) -> tuple[float, float]:
    """Return ``planet``'s predicted xy position after ``dt`` ticks.

    Only orbiting planets rotate; static planets and zero angular velocity keep their
    observed position.
    """
    if angular_velocity == 0.0 or not is_orbiting_planet(planet, center):
        return planet.x, planet.y

    cx, cy = center
    radius = distance_from_center(planet, center)
    current_angle = math.atan2(planet.y - cy, planet.x - cx)
    predicted_angle = current_angle + angular_velocity * dt
    return (cx + radius * math.cos(predicted_angle), cy + radius * math.sin(predicted_angle))


def moving_intercept_positions(
    source: Planet,
    target: Planet,
    angular_velocity: float,
    fleet_speed: float,
    center: tuple[float, float] = CENTER,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return predicted source and target positions for an intercept launch.

    The fixed-point solve uses the same future time for the launch origin and
    intercept point, so orbiting sources and orbiting targets are treated consistently.
    """
    if (
        angular_velocity == 0.0
        or fleet_speed <= 0.0
        or (not is_orbiting_planet(source, center) and not is_orbiting_planet(target, center))
    ):
        return (source.x, source.y), (target.x, target.y)

    travel_time = distance(source, target) / fleet_speed
    source_xy = (source.x, source.y)
    target_xy = (target.x, target.y)
    for _ in range(12):
        source_xy = future_orbit_position(source, travel_time, angular_velocity, center)
        target_xy = future_orbit_position(target, travel_time, angular_velocity, center)
        next_time = distance_xy(source_xy[0], source_xy[1], target_xy[0], target_xy[1]) / fleet_speed
        if math.isclose(next_time, travel_time, rel_tol=1e-6, abs_tol=1e-6):
            travel_time = next_time
            break
        travel_time = next_time

    return source_xy, target_xy


def sun_path_intersects(
    start: tuple[float, float],
    end: tuple[float, float],
    sun_center: tuple[float, float] = CENTER,
    sun_radius: float = SUN_RADIUS,
) -> bool:
    """Return whether the segment from ``start`` to ``end`` intersects the sun."""
    sx, sy = start
    ex, ey = end
    cx, cy = sun_center
    dx = ex - sx
    dy = ey - sy
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return distance_xy(sx, sy, cx, cy) <= sun_radius

    u = ((cx - sx) * dx + (cy - sy) * dy) / seg_len_sq
    u = min(1.0, max(0.0, u))
    closest_x = sx + u * dx
    closest_y = sy + u * dy
    return distance_xy(closest_x, closest_y, cx, cy) <= sun_radius


def launch_angle(
    source: Planet,
    target: Planet,
    angular_velocity: float,
    fleet_speed: float,
    center: tuple[float, float] = CENTER,
) -> float:
    """Return a launch angle that leads moving orbiting planets by predicted intercept time."""
    source_xy, target_xy = moving_intercept_positions(
        source, target, angular_velocity, fleet_speed, center
    )
    return math.atan2(target_xy[1] - source_xy[1], target_xy[0] - source_xy[0])


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
