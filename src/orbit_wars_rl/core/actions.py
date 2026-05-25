"""Convert model outputs into Orbit Wars launch actions."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

from .geometry import DEFAULT_SUN_COLLISION_RADIUS, LaunchSolution, predict_launch, trajectory_crosses_sun
from .types import Action, Planet

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ActionDecodeConfig:
    activation_threshold: float = 0.5
    reserve_ships: int = 1
    use_round: bool = False


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def get_fleet_speed(num_ships: int) -> float:
    """
    Return fleet speed based on fleet size.

    Speed ranges from 1.0 to 6.0.
    1 ship moves at 1.0.
    1000+ ships move at max speed.
    """
    min_speed = 1.0
    max_speed = 6.0

    if num_ships <= 1:
        return min_speed

    ships = min(num_ships, 1000)

    speed = min_speed + (max_speed - min_speed) * (
        math.log(ships) / math.log(1000)
    ) ** 1.5

    return float(min(max(speed, min_speed), max_speed))


SEND_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

def decode_model_outputs(
    source: Planet,
    candidates: Sequence[Planet],
    outputs: Sequence[float],
    config: ActionDecodeConfig | None = None,
    *,
    angular_velocity: float = 0.0,
    fleet_speed: float = 1.0,
    sun_radius: float = DEFAULT_SUN_COLLISION_RADIUS,
    precomputed_launches: Sequence[LaunchSolution | None] | None = None,
) -> list[Action]:
    cfg = config or ActionDecodeConfig()
    # New mode: [target_choice, amount_norm]
    if len(outputs) == 2:
        if not candidates:
            return []
        remaining = max(0, int(source.ships) - max(0, cfg.reserve_ships))
        if remaining <= 0:
            return []
        target_choice = int(outputs[0])
        if target_choice <= 0:
            return []
        idx = target_choice - 1
        if idx < 0 or idx >= min(4, len(candidates)):
            return []
        target = candidates[idx]
        min_send = min(10, remaining)
        amount_norm = clamp01(float(outputs[1]) / 100.0 if float(outputs[1]) > 1.0 else float(outputs[1]))
        span = max(0, remaining - min_send)
        ships = int(min_send + math.floor(amount_norm * span))
        ships = min(max(min_send, ships), remaining)
        launch = predict_launch(source, target, angular_velocity, get_fleet_speed(ships))
        if trajectory_crosses_sun(launch.source_xy, launch.target_xy, sun_radius=sun_radius):
            return []
        return [Action(source.id, launch.angle, ships)]

    output_len = len(outputs)
    if output_len == 9:
        output_len = 8
    if output_len not in (4, 8):
        raise ValueError(f"expected 4, 8, or 9 model outputs, got {len(outputs)}")

    remaining = max(0, int(source.ships) - max(0, cfg.reserve_ships))
    actions: list[Action] = []
    if output_len == 4:
        weights = [SEND_FRACTIONS[max(0, min(5, int(v)))] for v in outputs[: min(4, len(candidates))]]
        total_weight = sum(weights)
        if total_weight <= 0.0:
            return []
        requests = [int(math.floor(remaining * (w / total_weight))) if w > 0 else 0 for w in weights]
    else:
        requests = []
        for idx in range(min(4, len(candidates))):
            active = float(outputs[idx * 2]) > cfg.activation_threshold
            if not active:
                requests.append(0)
                continue
            pct = clamp01(outputs[idx * 2 + 1])
            requested = int(round(source.ships * pct)) if cfg.use_round else int(math.floor(source.ships * pct))
            requests.append(max(1, requested))

    for idx, requested in enumerate(requests):
        if remaining <= 0:
            break
        ships = min(max(0, requested), remaining)
        if ships <= 0:
            continue
        target = candidates[idx]
        # Candidate filtering can provide coarse precomputed launches, but the final
        # launch must be computed using the actual selected fleet size for this action.
        launch = predict_launch(source, target, angular_velocity, get_fleet_speed(ships))
        if trajectory_crosses_sun(launch.source_xy, launch.target_xy, sun_radius=sun_radius):
            LOGGER.debug("source=%s target=%s skipped sun-crossing launch", source.id, target.id)
            continue
        remaining -= ships
        actions.append(Action(source.id, launch.angle, ships))
    LOGGER.debug("source=%s decoded=%s", source.id, actions)
    return actions


def filter_candidates_with_valid_trajectories(
    source: Planet,
    candidates: Sequence[Planet],
    *,
    angular_velocity: float = 0.0,
    fleet_speed: float = 1.0,
    sun_radius: float = DEFAULT_SUN_COLLISION_RADIUS,
    max_candidates: int = 4,
) -> tuple[list[Planet], list[LaunchSolution]]:
    valid_candidates: list[Planet] = []
    launches: list[LaunchSolution] = []
    for target in candidates:
        if len(valid_candidates) >= max_candidates:
            break
        launch = predict_launch(source, target, angular_velocity, fleet_speed)
        if trajectory_crosses_sun(launch.source_xy, launch.target_xy, sun_radius=sun_radius):
            continue
        valid_candidates.append(target)
        launches.append(launch)
    return valid_candidates, launches
