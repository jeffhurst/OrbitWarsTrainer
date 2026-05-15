import math
from orbit_wars_rl.core.actions import ActionDecodeConfig, decode_model_outputs
from orbit_wars_rl.core.types import Planet


def planet(id, x, y, ships=10):
    return Planet(id, -1, x, y, 1, ships, 1)


def test_noop_overrides_other_outputs():
    src = planet(0, 0, 0)
    assert decode_model_outputs(src, [planet(1, 10, 0)], [1, 1, 0, 0, 0, 0, 0, 0, 0.6]) == []


def test_activation_clamp_reserve_and_sequential_allocation():
    src = planet(0, 0, 0, ships=10)
    targets = [planet(1, 10, 0), planet(2, 0, 10)]
    actions = decode_model_outputs(src, targets, [0.51, 0.6, 0.51, 1.5, 0, 0, 0, 0, 0], ActionDecodeConfig(reserve_ships=1))
    assert [a.num_ships for a in actions] == [6, 3]
    assert sum(a.num_ships for a in actions) == 9
    assert actions[0].to_row()[0] == 0
    assert math.isclose(actions[1].direction_angle, math.pi / 2)


def test_threshold_and_minimum_one_ship():
    src = planet(0, 0, 0, ships=4)
    actions = decode_model_outputs(src, [planet(1, 10, 0)], [0.5, 1, 0.51, 0.1, 0, 0, 0, 0, 0], ActionDecodeConfig(reserve_ships=1))
    assert actions == []
    actions = decode_model_outputs(src, [planet(1, 10, 0)], [0.51, 0.01, 0, 0, 0, 0, 0, 0, 0], ActionDecodeConfig(reserve_ships=1))
    assert actions[0].num_ships == 1
