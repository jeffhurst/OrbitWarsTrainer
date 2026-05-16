"""Convert 9 model outputs into Orbit Wars launch actions."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

from .geometry import launch_angle
from .types import Action, Planet

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ActionDecodeConfig:
    activation_threshold: float = 0.5
    reserve_ships: int = 1
    use_round: bool = False


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def decode_model_outputs(
    source: Planet,
    candidates: Sequence[Planet],
    outputs: Sequence[float],
    config: ActionDecodeConfig | None = None,
    *,
    angular_velocity: float = 0.0,
    fleet_speed: float = 1.0,
) -> list[Action]:
    cfg = config or ActionDecodeConfig()
    if len(outputs) != 9:
        raise ValueError(f"expected 9 model outputs, got {len(outputs)}")
    if float(outputs[8]) > cfg.activation_threshold:
        LOGGER.debug("source=%s no-op activated", source.id)
        return []

    remaining = max(0, int(source.ships) - max(0, cfg.reserve_ships))
    actions: list[Action] = []
    for idx in range(min(4, len(candidates))):
        active = float(outputs[idx * 2]) > cfg.activation_threshold
        if not active or remaining <= 0:
            continue
        pct = clamp01(outputs[idx * 2 + 1])
        requested = (
            int(round(source.ships * pct)) if cfg.use_round else int(math.floor(source.ships * pct))
        )
        ships = min(max(1, requested), remaining)
        if ships <= 0:
            continue
        remaining -= ships
        target = candidates[idx]
        actions.append(
            Action(source.id, launch_angle(source, target, angular_velocity, fleet_speed), ships)
        )
    LOGGER.debug("source=%s decoded=%s", source.id, actions)
    return actions
