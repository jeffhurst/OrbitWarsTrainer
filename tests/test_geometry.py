import math
from orbit_wars_rl.core.geometry import angle_between, distance
from orbit_wars_rl.core.types import Planet


def p(id, x, y):
    return Planet(id, -1, x, y, 1, 1, 1)


def test_distance_between_planets():
    assert distance(p(1, 0, 0), p(2, 3, 4)) == 5


def test_angle_convention_right_and_down():
    src = p(0, 10, 10)
    assert angle_between(src, p(1, 20, 10)) == 0
    assert math.isclose(angle_between(src, p(2, 10, 20)), math.pi / 2)


def test_launch_angle_static_target_uses_direct_angle():
    from orbit_wars_rl.core.geometry import launch_angle

    src = p(0, 10, 10)
    target = p(1, 99, 70)
    assert not math.isclose(target.x, 50)
    assert math.isclose(
        launch_angle(src, target, angular_velocity=0.25, fleet_speed=3.0),
        angle_between(src, target),
    )


def test_launch_angle_orbiting_target_leads_rotation():
    from orbit_wars_rl.core.geometry import launch_angle

    src = p(0, 50, 80)
    target = p(1, 70, 50)
    direct = angle_between(src, target)
    intercepted = launch_angle(src, target, angular_velocity=0.05, fleet_speed=5.0)
    assert not math.isclose(intercepted, direct)


def test_launch_angle_zero_angular_velocity_uses_direct_angle():
    from orbit_wars_rl.core.geometry import launch_angle

    src = p(0, 50, 80)
    target = p(1, 70, 50)
    assert math.isclose(
        launch_angle(src, target, angular_velocity=0.0, fleet_speed=5.0), angle_between(src, target)
    )
