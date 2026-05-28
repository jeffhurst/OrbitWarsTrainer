"""Aggressive but stable team-level reward utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from orbit_wars_rl.core.types import Planet, parse_fleets, parse_planets


@dataclass(frozen=True, slots=True)
class RewardShapingConfig:
    win_reward: float = 600.0
    loss_penalty: float = -1800.0
    timeout_min_reward: float = -60.0
    timeout_max_reward: float = 60.0
    fast_win_bonus: float = 300.0
    fast_win_reference_fraction: float = 0.4
    loss_survival_bonus: float = 0.0

    production_adv_weight: float = 55.0
    planet_adv_weight: float = 55.0
    ship_adv_weight: float = 45.0
    ship_delta_weight: float = 100.0
    production_delta_weight: float = 130.0
    net_capture_weight: float = 60.0

    enemy_capture_reward: float = 55.0
    neutral_capture_reward: float = 30.0
    captured_production_weight: float = 12.0

    local_action_weight: float = 1.0
    dense_reward_clip: float = 2.0
    reward_scale: float = 0.1
    loss_return_margin: float = 1.0


def _norm_diff(a: float, b: float, total: float) -> float:
    return (a - b) / max(1.0, total)


def player_score(obs: dict, player: int) -> float:
    planets = parse_planets(obs)
    fleets = parse_fleets(obs)
    return float(
        sum(p.ships for p in planets if p.owner == player)
        + sum(f.ships for f in fleets if f.owner == player)
    )


def production_advantage(obs: dict, player: int) -> float:
    planets = parse_planets(obs)
    enemy = 1 - player
    my_prod = sum(p.production for p in planets if p.owner == player)
    enemy_prod = sum(p.production for p in planets if p.owner == enemy)
    return _norm_diff(my_prod, enemy_prod, sum(p.production for p in planets))


def score_advantage(obs: dict, player: int) -> float:
    planets = parse_planets(obs)
    enemy = 1 - player
    return _norm_diff(
        float(sum(1 for p in planets if p.owner == player)),
        float(sum(1 for p in planets if p.owner == enemy)),
        float(len(planets)),
    )


def strategic_score(obs: dict, player: int, config: RewardShapingConfig | None = None) -> float:
    cfg = config or RewardShapingConfig()
    my_score = player_score(obs, player)
    enemy_score = player_score(obs, 1 - player)
    ship_adv = _norm_diff(my_score, enemy_score, my_score + enemy_score)
    return float(
        cfg.production_adv_weight * production_advantage(obs, player)
        + cfg.planet_adv_weight * score_advantage(obs, player)
        + cfg.ship_adv_weight * ship_adv
    )


def ships_sent_reward(actions: Iterable, config: RewardShapingConfig | None = None) -> float:
    del actions, config
    return 0.0


def win_speed_multiplier(turn_index: int, max_episode_turns: int, config: RewardShapingConfig | None = None) -> float:
    cfg = config or RewardShapingConfig()
    if max_episode_turns <= 0:
        return 1.0
    clamped_turn = min(max(1, int(turn_index)), int(max_episode_turns))
    remaining_turns = max_episode_turns - clamped_turn
    reference_turns = max(1.0, cfg.fast_win_reference_fraction * max_episode_turns)
    return float(2.0 ** (remaining_turns / reference_turns))


def win_speed_bonus(turn_index: int, max_episode_turns: int, config: RewardShapingConfig | None = None) -> float:
    """Backward-compatible additive fast-win bonus.

    Preserves the historical API used by PPO environment call sites while the
    new multiplier API is adopted.
    """
    cfg = config or RewardShapingConfig()
    return float(cfg.win_reward * (win_speed_multiplier(turn_index, max_episode_turns, cfg) - 1.0))


def game_outcome_reward(*, candidate_score: float, opponent_score: float, turn_index: int, max_episode_turns: int, config: RewardShapingConfig | None = None) -> float:
    cfg = config or RewardShapingConfig()
    if candidate_score > opponent_score:
        return float(cfg.win_reward * win_speed_multiplier(turn_index, max_episode_turns, cfg))
    if candidate_score < opponent_score:
        progress = min(max(turn_index, 1), max_episode_turns) / max(1, max_episode_turns)
        return float(cfg.loss_penalty + cfg.loss_survival_bonus * progress)
    return 0.0


def timeout_outcome_reward(candidate_score: float, opponent_score: float, config: RewardShapingConfig | None = None) -> float:
    cfg = config or RewardShapingConfig()
    total = max(1.0, abs(candidate_score) + abs(opponent_score))
    scaled = ((candidate_score - opponent_score) / total) * cfg.timeout_max_reward
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
