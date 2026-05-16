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


def future_orbit_position(
    planet: Planet,
    dt: float,
    angular_velocity: float,
    center: tuple[float, float] = CENTER,
) -> tuple[float, float]:
    """Return ``planet`` position after ``dt`` orbit updates around ``center``.

    Orbit Wars observations encode orbiting planets by their current Cartesian point.
    Assuming constant angular velocity, preserve the current orbital radius and advance
    only the polar angle by ``angular_velocity * dt``.
    """
    cx, cy = center
    radius = distance_from_center(planet, center)
    current_angle = math.atan2(planet.y - cy, planet.x - cx)
    predicted_angle = current_angle + angular_velocity * dt
    return (cx + radius * math.cos(predicted_angle), cy + radius * math.sin(predicted_angle))


def launch_angle(
    source: Planet,
    target: Planet,
    angular_velocity: float,
    fleet_speed: float,
    center: tuple[float, float] = CENTER,
    launch_delay: float = 1.0,
) -> float:
    """Return a launch angle that leads orbiting endpoints by intercept time.

    Orbit Wars advances orbiting planets once before applying launch actions for the
    current turn, so the fleet origin for an orbiting source is the source's position
    after ``launch_delay`` orbit updates. Orbiting targets are then led from that
    predicted origin for the fleet travel time. Static endpoints preserve the direct
    ``atan2`` behavior.
    """
    source_orbiting = is_orbiting_planet(source, center)
    target_orbiting = is_orbiting_planet(target, center)
    should_predict_orbits = (source_orbiting or target_orbiting) and angular_velocity != 0.0

    if not should_predict_orbits:
        return angle_between(source, target)

    launch_x, launch_y = (
        future_orbit_position(source, launch_delay, angular_velocity, center)
        if source_orbiting
        else (source.x, source.y)
    )

    def predicted_target_position(flight_time: float) -> tuple[float, float]:
        if not target_orbiting:
            return (target.x, target.y)
        return future_orbit_position(target, launch_delay + flight_time, angular_velocity, center)

    if target_orbiting and fleet_speed > 0.0:
        target_x, target_y = future_orbit_position(target, launch_delay, angular_velocity, center)
        t = distance_xy(launch_x, launch_y, target_x, target_y) / fleet_speed
        for _ in range(12):
            predicted_x, predicted_y = predicted_target_position(t)
            next_t = distance_xy(launch_x, launch_y, predicted_x, predicted_y) / fleet_speed
            if math.isclose(next_t, t, rel_tol=1e-6, abs_tol=1e-6):
                t = next_t
                break
            t = next_t
    else:
        t = 0.0

    predicted_x, predicted_y = predicted_target_position(t)
    return math.atan2(predicted_y - launch_y, predicted_x - launch_x)


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
