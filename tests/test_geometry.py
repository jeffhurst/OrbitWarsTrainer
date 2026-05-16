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


def test_segment_intersects_circle_direct_hit_and_miss():
    from orbit_wars_rl.core.geometry import segment_intersects_circle, trajectory_crosses_sun

    assert segment_intersects_circle(40, 50, 60, 50, 50, 50, 5)
    assert trajectory_crosses_sun((40, 50), (60, 50), sun_radius=5)
    assert not segment_intersects_circle(40, 56, 60, 56, 50, 50, 5)
    assert not trajectory_crosses_sun((40, 56), (60, 56), sun_radius=5)


def test_launch_angle_orbiting_source_static_target_uses_current_source_position():
    from orbit_wars_rl.core.geometry import launch_angle

    src = p(0, 70, 50)
    target = p(1, 99, 70)

    assert math.isclose(
        launch_angle(src, target, angular_velocity=0.25, fleet_speed=3.0),
        angle_between(src, target),
    )


def test_predict_launch_returns_current_source_and_moving_target_endpoint_segment():
    from orbit_wars_rl.core.geometry import predict_launch, trajectory_crosses_sun

    src = p(0, 0, 0)
    target = p(1, 45, 60)

    assert trajectory_crosses_sun((src.x, src.y), (target.x, target.y))
    launch = predict_launch(src, target, angular_velocity=0.05, fleet_speed=2.0)

    assert launch.source_xy == (src.x, src.y)
    assert not math.isclose(launch.target_xy[1], target.y)
    assert not trajectory_crosses_sun(launch.source_xy, launch.target_xy)
