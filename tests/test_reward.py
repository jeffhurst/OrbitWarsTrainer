import pytest

from orbit_wars_rl.core.types import Planet
from orbit_wars_rl.training.reward import RewardShapingConfig, planet_capture_reward, ships_sent_reward


def planet(planet_id, owner, production):
    return Planet(planet_id, owner, 0.0, 0.0, 1.0, 10, production)


def test_ships_sent_reward_is_tiny_per_launched_ship():
    assert ships_sent_reward([[0, 0.0, 3], [1, 1.0, 2]]) == pytest.approx(0.005)


def test_planet_capture_reward_scales_with_production_and_enemy_ownership():
    before = [planet(1, -1, 5.0), planet(2, 1, 7.0)]
    after = [planet(1, 0, 5.0), planet(2, 0, 7.0)]

    reward = planet_capture_reward(before, after, player=0)

    assert reward == pytest.approx(5.0 * 0.10 + 7.0 * 0.50)


def test_planet_capture_reward_penalizes_losing_production():
    cfg = RewardShapingConfig(loss_production_penalty=0.25)
    before = [planet(1, 0, 6.0)]
    after = [planet(1, 1, 6.0)]

    reward = planet_capture_reward(before, after, player=0, config=cfg)

    assert reward == pytest.approx(-1.5)
