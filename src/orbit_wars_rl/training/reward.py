"""Simple reward utilities for rollout/evaluation bookkeeping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from orbit_wars_rl.core.types import Planet, parse_fleets, parse_planets


@dataclass(frozen=True, slots=True)
class RewardShapingConfig:
    """Small auxiliary reward terms that keep production control dominant."""

    send_ship_reward: float = 0.0
    capture_production_bonus: float = 0.10
    enemy_capture_production_bonus: float = 0.50
    loss_production_penalty: float = 0.10
    win_base_bonus: float = 25.0
    fastest_win_bonus: float = 75.0
    loss_game_penalty: float = 10.0


def player_score(obs: dict, player: int) -> float:
    """Return the terminal outcome score used by training and evaluation.

    This trainer scores terminal outcomes by total ship count controlled by each player,
    including ships stationed on planets and ships in flight. Production control remains a dense
    shaping signal, but terminal win/loss rewards and evaluation win rate intentionally use this
    shared ship-count outcome.
    """
    planets = parse_planets(obs)
    fleets = parse_fleets(obs)
    planet_ships = sum(p.ships for p in planets if p.owner == player)
    fleet_ships = sum(f.ships for f in fleets if f.owner == player)
    return float(planet_ships + fleet_ships)


def production_controlled(obs: dict, player: int) -> float:
    return float(sum(p.production for p in parse_planets(obs) if p.owner == player))




def production_advantage(obs: dict, player: int) -> float:
    opponent = 1 - player
    return production_controlled(obs, player) - production_controlled(obs, opponent)


def score_advantage(obs: dict, player: int) -> float:
    opponent = 1 - player
    return player_score(obs, player) - player_score(obs, opponent)


def ships_sent_reward(actions: Iterable, config: RewardShapingConfig | None = None) -> float:
    """Return a tiny reward for launching ships instead of no-oping."""
    cfg = config or RewardShapingConfig()
    ships_sent = 0
    for action in actions:
        if hasattr(action, "num_ships"):
            ships_sent += max(0, int(action.num_ships))
        else:
            ships_sent += max(0, int(action[2]))
    return float(ships_sent * cfg.send_ship_reward)


def win_speed_bonus(turn_index: int, max_episode_turns: int, config: RewardShapingConfig | None = None) -> float:
    """Return a terminal win bonus that is larger when the game is won earlier."""
    cfg = config or RewardShapingConfig()
    if max_episode_turns <= 0:
        remaining_fraction = 0.0
    else:
        clamped_turn = min(max(1, int(turn_index)), int(max_episode_turns))
        remaining_fraction = (max_episode_turns - clamped_turn) / max_episode_turns
    return float(cfg.win_base_bonus + cfg.fastest_win_bonus * remaining_fraction)


def game_outcome_reward(
    *,
    candidate_score: float,
    opponent_score: float,
    turn_index: int,
    max_episode_turns: int,
    config: RewardShapingConfig | None = None,
) -> float:
    """Return terminal game outcome shaping: speed-scaled win bonus, small static loss.

    Draws and non-terminal equal-score outcomes return zero. The loss penalty intentionally does
    not scale by turn count so losing late is not rewarded relative to losing early.
    """
    cfg = config or RewardShapingConfig()
    if candidate_score > opponent_score:
        return win_speed_bonus(turn_index, max_episode_turns, cfg)
    if candidate_score < opponent_score:
        return float(-cfg.loss_game_penalty)
    return 0.0


def planet_capture_reward(
    before_planets: Iterable[Planet],
    after_planets: Iterable[Planet],
    player: int,
    config: RewardShapingConfig | None = None,
) -> float:
    """Reward production-scaled planet captures and penalize production-scaled losses.

    The primary reward remains the raw controlled-production delta. These shaping terms are
    deliberately smaller multipliers on planet production; capturing an enemy planet receives the
    largest auxiliary bonus because it both adds to the candidate and denies the opponent.
    """
    cfg = config or RewardShapingConfig()
    before_by_id = {p.id: p for p in before_planets}
    reward = 0.0
    for after in after_planets:
        before = before_by_id.get(after.id)
        if before is None or before.owner == after.owner:
            continue
        if after.owner == player and before.owner != player:
            multiplier = cfg.enemy_capture_production_bonus if before.owner >= 0 else cfg.capture_production_bonus
            reward += after.production * multiplier
        elif before.owner == player and after.owner != player:
            reward -= before.production * cfg.loss_production_penalty
    return float(reward)
