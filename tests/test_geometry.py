import math

from orbit_wars_rl.core.geometry import (
    angle_between,
    distance,
    distance_xy,
    future_orbit_position,
    launch_angle,
)
from orbit_wars_rl.core.types import Planet


def p(id, x, y):
    return Planet(id, -1, x, y, 1, 1, 1)


def test_distance_between_planets():
    assert distance(p(1, 0, 0), p(2, 3, 4)) == 5


def test_angle_convention_right_and_down():
    src = p(0, 10, 10)
    assert angle_between(src, p(1, 20, 10)) == 0
    assert math.isclose(angle_between(src, p(2, 10, 20)), math.pi / 2)


def test_launch_angle_static_source_to_static_target_unchanged():
    src = p(0, 10, 10)
    target = p(1, 99, 70)
    assert math.isclose(
        launch_angle(src, target, angular_velocity=0.25, fleet_speed=3.0),
        angle_between(src, target),
    )


def test_launch_angle_orbiting_source_to_static_target_uses_predicted_origin():
    src = p(0, 70, 50)
    target = p(1, 99, 70)
    direct = angle_between(src, target)

    launch_x, launch_y = future_orbit_position(src, dt=1.0, angular_velocity=0.1)
    expected = math.atan2(target.y - launch_y, target.x - launch_x)

    intercepted = launch_angle(src, target, angular_velocity=0.1, fleet_speed=5.0)
    assert not math.isclose(intercepted, direct)
    assert math.isclose(intercepted, expected)


def test_launch_angle_orbiting_source_to_orbiting_target_uses_both_predicted_positions():
    src = p(0, 70, 50)
    target = p(1, 50, 70)
    angular_velocity = 0.05
    fleet_speed = 5.0

    launch_x, launch_y = future_orbit_position(src, dt=1.0, angular_velocity=angular_velocity)
    t = distance_xy(
        launch_x,
        launch_y,
        *future_orbit_position(target, dt=1.0, angular_velocity=angular_velocity),
    ) / fleet_speed
    for _ in range(12):
        target_x, target_y = future_orbit_position(target, dt=1.0 + t, angular_velocity=angular_velocity)
        next_t = distance_xy(launch_x, launch_y, target_x, target_y) / fleet_speed
        if math.isclose(next_t, t, rel_tol=1e-6, abs_tol=1e-6):
            t = next_t
            break
        t = next_t

    predicted_target_x, predicted_target_y = future_orbit_position(
        target, dt=1.0 + t, angular_velocity=angular_velocity
    )
    expected = math.atan2(predicted_target_y - launch_y, predicted_target_x - launch_x)

    direct = angle_between(src, target)
    intercepted = launch_angle(src, target, angular_velocity=angular_velocity, fleet_speed=fleet_speed)
    assert not math.isclose(intercepted, direct)
    assert math.isclose(intercepted, expected)


def test_launch_angle_orbiting_target_leads_rotation():
    src = p(0, 50, 80)
    target = p(1, 70, 50)
    direct = angle_between(src, target)
    intercepted = launch_angle(src, target, angular_velocity=0.05, fleet_speed=5.0)
    assert not math.isclose(intercepted, direct)


def test_launch_angle_zero_angular_velocity_uses_direct_angle():
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
