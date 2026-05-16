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
DEFAULT_SUN_COLLISION_RADIUS = 10.0


@dataclass(frozen=True, slots=True)
class QuadrantConfig:
    center: tuple[float, float] = CENTER
    ccw_map: dict[int, int] | None = None

    def next_ccw(self, quadrant: int) -> int:
        mapping = self.ccw_map or {1: 2, 2: 3, 3: 4, 4: 1}
        return mapping[quadrant]


@dataclass(frozen=True, slots=True)
class LaunchSolution:
    """Predicted straight launch segment and corresponding launch angle."""

    angle: float
    source_xy: tuple[float, float]
    target_xy: tuple[float, float]


def distance_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def distance(a: Planet, b: Planet) -> float:
    return distance_xy(a.x, a.y, b.x, b.y)


def distance_from_center(p: Planet, center: tuple[float, float] = CENTER) -> float:
    return distance_xy(p.x, p.y, center[0], center[1])


def angle_between(source: Planet, target: Planet) -> float:
    """Return launch angle where 0=right and pi/2=down."""
    return math.atan2(target.y - source.y, target.x - source.x)


def segment_intersects_circle(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    radius: float,
) -> bool:
    """Return whether segment AB touches or crosses a circle."""
    if radius < 0:
        raise ValueError("circle radius must be non-negative")

    abx = bx - ax
    aby = by - ay
    length_squared = abx * abx + aby * aby
    if length_squared == 0.0:
        return distance_xy(ax, ay, cx, cy) <= radius

    t = ((cx - ax) * abx + (cy - ay) * aby) / length_squared
    t = max(0.0, min(1.0, t))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return distance_xy(closest_x, closest_y, cx, cy) <= radius


def trajectory_crosses_sun(
    source_xy: tuple[float, float],
    target_xy: tuple[float, float],
    sun_center: tuple[float, float] = CENTER,
    sun_radius: float = DEFAULT_SUN_COLLISION_RADIUS,
) -> bool:
    """Return whether a predicted straight fleet path intersects the sun."""
    return segment_intersects_circle(
        source_xy[0],
        source_xy[1],
        target_xy[0],
        target_xy[1],
        sun_center[0],
        sun_center[1],
        sun_radius,
    )


def sun_collision_radius_from_obs(
    obs: object,
    default: float = DEFAULT_SUN_COLLISION_RADIUS,
) -> float:
    """Read the sun collision radius from an observation/config when available.

    Current local fixtures do not expose an official value, so callers fall back to the clearly
    named ``DEFAULT_SUN_COLLISION_RADIUS``. Several likely key names are supported to make this
    helper compatible with richer environments without changing agent code.
    """
    keys = ("sun_collision_radius", "sun_radius")
    if hasattr(obs, "get"):
        for key in keys:
            value = obs.get(key)  # type: ignore[union-attr]
            if value is not None:
                return float(value)
        config = obs.get("configuration") or obs.get("config")  # type: ignore[union-attr]
        if hasattr(config, "get"):
            for key in keys:
                value = config.get(key)
                if value is not None:
                    return float(value)
    for key in keys:
        value = getattr(obs, key, None)
        if value is not None:
            return float(value)
    config = getattr(obs, "configuration", None) or getattr(obs, "config", None)
    for key in keys:
        value = getattr(config, key, None)
        if value is not None:
            return float(value)
    return float(default)


def predicted_planet_position(
    planet: Planet,
    t: float,
    angular_velocity: float,
    center: tuple[float, float] = CENTER,
) -> tuple[float, float]:
    """Return a planet's predicted position after ``t`` time units."""
    if angular_velocity == 0.0 or not is_orbiting_planet(planet, center):
        return (planet.x, planet.y)

    cx, cy = center
    radius = distance_from_center(planet, center)
    current_angle = math.atan2(planet.y - cy, planet.x - cx)
    predicted_angle = current_angle + angular_velocity * t
    return (cx + radius * math.cos(predicted_angle), cy + radius * math.sin(predicted_angle))

def predict_launch(
    source: Planet,
    target: Planet,
    angular_velocity: float,
    fleet_speed: float,
    center: tuple[float, float] = CENTER,
) -> LaunchSolution:
    """Return a launch angle that intercepts an orbiting target.

    Static targets use direct atan2. Orbiting targets are solved by finding
    the earliest time t where:

        distance(source, predicted_target_position(t)) - source.radius == fleet_speed * t
    """

    source_xy = (source.x, source.y)

    if (
        not is_orbiting_planet(target, center)
        or angular_velocity == 0.0
        or fleet_speed <= 0.0
    ):
        target_xy = (target.x, target.y)
        return LaunchSolution(angle_between(source, target), source_xy, target_xy)

    def intercept_error(t: float) -> float:
        tx, ty = predicted_planet_position(target, t, angular_velocity, center)
        d = distance_xy(source.x, source.y, tx, ty) - source.radius
        return d - fleet_speed * t

    # Choose scan resolution.
    # Smaller values are more accurate but slower.
    scan_dt = 0.25

    # Pick a reasonable max time horizon.
    # You can tune this for your game.
    max_t = 500.0

    prev_t = 0.0
    prev_f = intercept_error(prev_t)

    t = scan_dt

    while t <= max_t:
        curr_f = intercept_error(t)

        # We found a sign crossing, so the intercept time is between prev_t and t.
        if prev_f * curr_f <= 0.0:
            lo = prev_t
            hi = t

            # Bisection refinement
            for _ in range(40):
                mid = (lo + hi) * 0.5
                mid_f = intercept_error(mid)

                if prev_f * mid_f <= 0.0:
                    hi = mid
                    curr_f = mid_f
                else:
                    lo = mid
                    prev_f = mid_f

            intercept_t = (lo + hi) * 0.5
            target_xy = predicted_planet_position(
                target,
                intercept_t,
                angular_velocity,
                center,
            )

            angle = math.atan2(
                target_xy[1] - source.y,
                target_xy[0] - source.x,
            )

            return LaunchSolution(angle, source_xy, target_xy)

        prev_t = t
        prev_f = curr_f
        t += scan_dt

    # Fallback: if no bracket was found, aim at current position.
    # But this should be rare if max_t is large enough.
    target_xy = (target.x, target.y)
    return LaunchSolution(
        math.atan2(target.y - source.y, target.x - source.x),
        source_xy,
        target_xy,
    )

def launch_angle(
    source: Planet,
    target: Planet,
    angular_velocity: float,
    fleet_speed: float,
    center: tuple[float, float] = CENTER,
) -> float:
    """Return a launch angle that leads orbiting endpoints by predicted intercept time."""
    return predict_launch(source, target, angular_velocity, fleet_speed, center).angle


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
