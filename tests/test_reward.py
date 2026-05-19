import pytest

from orbit_wars_rl.core.types import Planet
from orbit_wars_rl.training.reward import (
    RewardShapingConfig,
    action_targeting_reward,
    game_outcome_reward,
    planet_capture_reward,
    strategic_score,
    timeout_outcome_reward,
)


def planet(planet_id, owner, ships, production):
    return Planet(planet_id, owner, 0.0, 0.0, 1.0, ships, production)


def test_win_reward_is_strongly_positive_and_faster_is_better():
    cfg = RewardShapingConfig()
    early = game_outcome_reward(candidate_score=100, opponent_score=10, turn_index=10, max_episode_turns=100, config=cfg)
    late = game_outcome_reward(candidate_score=100, opponent_score=10, turn_index=90, max_episode_turns=100, config=cfg)
    assert early > 200
    assert late > 200
    assert early > late


def test_loss_reward_is_strongly_negative_and_slow_loss_is_less_negative():
    cfg = RewardShapingConfig()
    early = game_outcome_reward(candidate_score=10, opponent_score=100, turn_index=10, max_episode_turns=100, config=cfg)
    late = game_outcome_reward(candidate_score=10, opponent_score=100, turn_index=90, max_episode_turns=100, config=cfg)
    assert early < -150
    assert late < 0
    assert early < late


def test_timeout_reward_bounded_small():
    cfg = RewardShapingConfig()
    reward = timeout_outcome_reward(120, 100, cfg)
    assert -60.0 <= reward <= 60.0


def test_capture_enemy_more_than_neutral():
    before = [planet(1, -1, 10, 5.0), planet(2, 1, 10, 5.0)]
    after_enemy = [planet(1, -1, 10, 5.0), planet(2, 0, 10, 5.0)]
    after_neutral = [planet(1, 0, 10, 5.0), planet(2, 1, 10, 5.0)]
    enemy_reward = planet_capture_reward(before, after_enemy, player=0)
    neutral_reward = planet_capture_reward(before, after_neutral, player=0)
    assert enemy_reward > neutral_reward


def test_strategic_score_favors_production_and_planets_over_ships_alone():
    base = {"planets": [[0, 0, 0.0, 0.0, 1.0, 10, 1.0], [1, 1, 0.0, 0.0, 1.0, 10, 1.0]], "fleets": []}
    prod_gain = {"planets": [[0, 0, 0.0, 0.0, 1.0, 10, 4.0], [1, 1, 0.0, 0.0, 1.0, 10, 1.0]], "fleets": []}
    ship_gain = {"planets": [[0, 0, 0.0, 0.0, 1.0, 40, 1.0], [1, 1, 0.0, 0.0, 1.0, 10, 1.0]], "fleets": []}
    base_score = strategic_score(base, 0)
    assert strategic_score(prod_gain, 0) - base_score > strategic_score(ship_gain, 0) - base_score


def test_missing_data_does_not_crash_reward_calculation():
    assert strategic_score({}, 0) == pytest.approx(0.0)


def test_action_targeting_reward_prefers_overmatch_and_low_ship_target():
    candidates = [
        planet(1, -1, 12, 2.0),
        planet(2, -1, 4, 2.0),
        planet(3, -1, 20, 2.0),
        planet(4, -1, 8, 2.0),
    ]
    low_target = action_targeting_reward(source_ships=30, candidates=candidates, action_values=[0, 5, 0, 0])
    high_target = action_targeting_reward(source_ships=30, candidates=candidates, action_values=[5, 0, 0, 0])
    assert low_target > 0
    assert low_target > high_target
