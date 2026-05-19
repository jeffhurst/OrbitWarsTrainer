"""Aggressive but stable team-level reward utilities.

Reward philosophy:
- Terminal outcomes dominate all shaping.
- Dense shaping rewards state improvements (deltas), not static advantage.
- Captures and pressure are rewarded when they improve winning chances.
- Local per-source action shaping is tiny compared to team rewards.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from orbit_wars_rl.core.types import Planet, parse_fleets, parse_planets


@dataclass(frozen=True, slots=True)
class RewardShapingConfig:
    """Configurable constants for aggressive reward shaping."""

    win_reward: float = 1000.0
    loss_penalty: float = -1000.0
    timeout_min_reward: float = -100.0
    timeout_max_reward: float = 100.0
    fast_win_bonus: float = 300.0

    production_adv_weight: float = 80.0
    planet_adv_weight: float = 50.0
    ship_adv_weight: float = 25.0
    pressure_adv_weight: float = 20.0

    enemy_capture_reward: float = 30.0
    neutral_capture_reward: float = 15.0
    captured_production_weight: float = 5.0

    local_action_weight: float = 0.05
    successful_enemy_target_bonus: float = 1.0
    valuable_neutral_target_bonus: float = 0.6
    suicidal_attack_penalty: float = 1.2

    idle_ship_ratio_threshold: float = 0.60
    idle_ship_penalty: float = 8.0
    min_pressure_ratio_for_no_idle_penalty: float = 0.10


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm_diff(a: float, b: float, total: float) -> float:
    denom = max(1.0, total)
    return (a - b) / denom


def _planet_counts(planets: Iterable[Planet], player: int) -> tuple[int, int, int]:
    enemy = 1 - player
    my_count = sum(1 for p in planets if p.owner == player)
    enemy_count = sum(1 for p in planets if p.owner == enemy)
    total = len(list(planets)) if not isinstance(planets, list) else len(planets)
    return my_count, enemy_count, total


def player_score(obs: dict, player: int) -> float:
    planets = parse_planets(obs)
    fleets = parse_fleets(obs)
    planet_ships = sum(p.ships for p in planets if p.owner == player)
    fleet_ships = sum(f.ships for f in fleets if f.owner == player)
    return float(planet_ships + fleet_ships)


def production_controlled(obs: dict, player: int) -> float:
    return float(sum(p.production for p in parse_planets(obs) if p.owner == player))


def production_advantage(obs: dict, player: int) -> float:
    planets = parse_planets(obs)
    enemy = 1 - player
    my_prod = sum(p.production for p in planets if p.owner == player)
    enemy_prod = sum(p.production for p in planets if p.owner == enemy)
    total_prod = sum(p.production for p in planets)
    return _norm_diff(my_prod, enemy_prod, total_prod)


def score_advantage(obs: dict, player: int) -> float:
    planets = parse_planets(obs)
    enemy = 1 - player
    my_planets = sum(1 for p in planets if p.owner == player)
    enemy_planets = sum(1 for p in planets if p.owner == enemy)
    total_planets = len(planets)
    return _norm_diff(float(my_planets), float(enemy_planets), float(total_planets))


def fleet_pressure_advantage(obs: dict, player: int) -> float:
    fleets = parse_fleets(obs)
    enemy = 1 - player
    my_pressure = sum(f.ships for f in fleets if f.owner == player)
    enemy_pressure = sum(f.ships for f in fleets if f.owner == enemy)
    total_fleet_ships = sum(f.ships for f in fleets)
    return _norm_diff(my_pressure, enemy_pressure, total_fleet_ships)


def strategic_score(obs: dict, player: int, config: RewardShapingConfig | None = None) -> float:
    cfg = config or RewardShapingConfig()
    ship_adv = _norm_diff(
        player_score(obs, player),
        player_score(obs, 1 - player),
        player_score(obs, player) + player_score(obs, 1 - player),
    )
    return float(
        cfg.production_adv_weight * production_advantage(obs, player)
        + cfg.planet_adv_weight * score_advantage(obs, player)
        + cfg.ship_adv_weight * ship_adv
        + cfg.pressure_adv_weight * fleet_pressure_advantage(obs, player)
    )


def ships_sent_reward(actions: Iterable, config: RewardShapingConfig | None = None) -> float:
    # Preserved for compatibility; aggressive reward does not directly reward sending ships.
    del actions, config
    return 0.0


def win_speed_bonus(turn_index: int, max_episode_turns: int, config: RewardShapingConfig | None = None) -> float:
    cfg = config or RewardShapingConfig()
    if max_episode_turns <= 0:
        return 0.0
    clamped_turn = min(max(1, int(turn_index)), int(max_episode_turns))
    remaining_fraction = (max_episode_turns - clamped_turn) / max_episode_turns
    return float(cfg.fast_win_bonus * remaining_fraction)


def game_outcome_reward(*, candidate_score: float, opponent_score: float, turn_index: int, max_episode_turns: int, config: RewardShapingConfig | None = None) -> float:
    cfg = config or RewardShapingConfig()
    if candidate_score > opponent_score:
        return float(cfg.win_reward + win_speed_bonus(turn_index, max_episode_turns, cfg))
    if candidate_score < opponent_score:
        progress = min(max(turn_index, 1), max_episode_turns) / max(1, max_episode_turns)
        return float(cfg.loss_penalty + 800.0 * progress)
    return 0.0


def timeout_outcome_reward(candidate_score: float, opponent_score: float, config: RewardShapingConfig | None = None) -> float:
    cfg = config or RewardShapingConfig()
    total = max(1.0, abs(candidate_score) + abs(opponent_score))
    normalized = (candidate_score - opponent_score) / total
    scaled = normalized * cfg.timeout_max_reward
    return float(max(cfg.timeout_min_reward, min(cfg.timeout_max_reward, scaled)))


def planet_capture_reward(before_planets: Iterable[Planet], after_planets: Iterable[Planet], player: int, config: RewardShapingConfig | None = None) -> float:
    cfg = config or RewardShapingConfig()
    before_by_id = {p.id: p for p in before_planets}
    reward = 0.0
    for after in after_planets:
        before = before_by_id.get(after.id)
        if before is None or before.owner == after.owner:
            continue
        if after.owner == player and before.owner != player:
            reward += cfg.enemy_capture_reward if before.owner >= 0 else cfg.neutral_capture_reward
            reward += cfg.captured_production_weight * after.production
    return float(reward)


def idle_ship_penalty(obs: dict, player: int, config: RewardShapingConfig | None = None) -> float:
    cfg = config or RewardShapingConfig()
    my_prod_adv = production_advantage(obs, player)
    if my_prod_adv > 0:
        return 0.0
    planets = parse_planets(obs)
    my_planet_ships = sum(p.ships for p in planets if p.owner == player)
    my_total_ships = player_score(obs, player)
    if my_total_ships <= 0:
        return 0.0
    idle_ratio = my_planet_ships / my_total_ships
    pressure = fleet_pressure_advantage(obs, player)
    if idle_ratio > cfg.idle_ship_ratio_threshold and pressure < cfg.min_pressure_ratio_for_no_idle_penalty:
        return float(-cfg.idle_ship_penalty * (idle_ratio - cfg.idle_ship_ratio_threshold))
    return 0.0
