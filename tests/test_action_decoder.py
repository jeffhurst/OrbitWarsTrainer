import math
from orbit_wars_rl.core.actions import ActionDecodeConfig, decode_model_outputs
from orbit_wars_rl.core.types import Planet


def planet(id, x, y, ships=10):
    return Planet(id, -1, x, y, 1, ships, 1)


def test_activation_clamp_reserve_and_sequential_allocation():
    src = planet(0, 0, 0, ships=10)
    targets = [planet(1, 10, 0), planet(2, 0, 10)]
    actions = decode_model_outputs(
        src, targets, [0.51, 0.6, 0.51, 1.5, 0, 0, 0, 0], ActionDecodeConfig(reserve_ships=1)
    )
    assert [a.num_ships for a in actions] == [6, 3]
    assert sum(a.num_ships for a in actions) == 9
    assert actions[0].to_row()[0] == 0
    assert math.isclose(actions[1].direction_angle, math.pi / 2)


def test_threshold_and_minimum_one_ship():
    src = planet(0, 0, 0, ships=4)
    actions = decode_model_outputs(
        src,
        [planet(1, 10, 0)],
        [0.5, 1, 0.51, 0.1, 0, 0, 0, 0],
        ActionDecodeConfig(reserve_ships=1),
    )
    assert actions == []
    actions = decode_model_outputs(
        src,
        [planet(1, 10, 0)],
        [0.51, 0.01, 0, 0, 0, 0, 0, 0],
        ActionDecodeConfig(reserve_ships=1),
    )
    assert actions[0].num_ships == 1


def test_decoder_uses_intercept_aware_launch_angle_context():
    src = planet(0, 50, 80, ships=10)
    target = planet(1, 70, 50)
    outputs = [0.51, 0.5, 0, 0, 0, 0, 0, 0]
    direct_actions = decode_model_outputs(
        src, [target], outputs, angular_velocity=0.0, fleet_speed=5.0
    )
    intercept_actions = decode_model_outputs(
        src, [target], outputs, angular_velocity=0.05, fleet_speed=5.0
    )
    assert math.isclose(
        direct_actions[0].direction_angle, math.atan2(target.y - src.y, target.x - src.x)
    )
    assert not math.isclose(intercept_actions[0].direction_angle, direct_actions[0].direction_angle)


def test_decoder_skips_launch_segment_passing_through_sun():
    src = planet(0, 40, 50, ships=10)
    target = planet(1, 60, 50)

    actions = decode_model_outputs(
        src,
        [target],
        [0.51, 0.5, 0, 0, 0, 0, 0, 0],
        ActionDecodeConfig(reserve_ships=1),
        sun_radius=5,
    )

    assert actions == []


def test_decoder_allows_launch_segment_outside_sun_radius():
    src = planet(0, 40, 56, ships=10)
    target = planet(1, 60, 56)

    actions = decode_model_outputs(
        src,
        [target],
        [0.51, 0.5, 0, 0, 0, 0, 0, 0],
        ActionDecodeConfig(reserve_ships=1),
        sun_radius=5,
    )

    assert len(actions) == 1


def test_decoder_filters_using_predicted_segment_not_current_positions():
    src = planet(0, 0, 0, ships=10)
    target = planet(1, 45, 60)

    actions = decode_model_outputs(
        src,
        [target],
        [0.51, 0.5, 0, 0, 0, 0, 0, 0],
        ActionDecodeConfig(reserve_ships=1),
        angular_velocity=0.05,
        fleet_speed=2.0,
        sun_radius=5,
    )

    assert len(actions) == 1


def test_discrete_multitarget_weights():
    src = planet(0, 0, 0, ships=21)
    targets = [planet(1, 10, 0), planet(2, 0, 10), planet(3, -10, 0), planet(4, 0, -10)]
    actions = decode_model_outputs(src, targets, [5, 5, 0, 0], ActionDecodeConfig(reserve_ships=1))
    assert len(actions) == 2
    assert sum(a.num_ships for a in actions) <= 20
    assert all(a.num_ships > 0 for a in actions)


def test_legacy_nine_output_vector_remains_supported():
    src = planet(0, 0, 0, ships=10)
    targets = [planet(1, 10, 0)]
    actions = decode_model_outputs(
        src,
        targets,
        [0.51, 0.5, 0, 0, 0, 0, 0, 0, 0.25],
        ActionDecodeConfig(reserve_ships=1),
    )
    assert len(actions) == 1
    assert actions[0].num_ships == 5
