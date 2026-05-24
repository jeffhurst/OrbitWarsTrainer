import math

from orbit_wars_rl.core.geometry import (
    angle_between,
    distance,
    distance_xy,
    launch_angle,
    predict_launch,
    predicted_planet_position,
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


def test_launch_angle_orbiting_source_to_static_target_uses_current_source_position():
    src = p(0, 70, 50)
    target = p(1, 99, 70)

    assert math.isclose(
        launch_angle(src, target, angular_velocity=0.25, fleet_speed=3.0),
        angle_between(src, target),
    )


def test_launch_angle_current_source_to_orbiting_target_leads_rotation():
    src = p(0, 10, 30)
    target = p(1, 50, 45)
    angular_velocity = 0.05
    fleet_speed = 2.0

    t = (distance_xy(src.x, src.y, target.x, target.y) - src.radius - target.radius) / fleet_speed
    for _ in range(12):
        predicted_target_x, predicted_target_y = predicted_planet_position(
            target, t, angular_velocity
        )
        next_t = (
            distance_xy(src.x, src.y, predicted_target_x, predicted_target_y) - src.radius - target.radius
        ) / fleet_speed
        if math.isclose(next_t, t, rel_tol=1e-6, abs_tol=1e-6):
            t = next_t
            break
        t = next_t

    predicted_target_x, predicted_target_y = predicted_planet_position(target, t, angular_velocity)
    expected = math.atan2(predicted_target_y - src.y, predicted_target_x - src.x)

    intercepted = launch_angle(src, target, angular_velocity=angular_velocity, fleet_speed=fleet_speed)
    assert not math.isclose(intercepted, angle_between(src, target))
    assert math.isclose(intercepted, expected, rel_tol=1e-6, abs_tol=1e-6)


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


def test_predict_launch_returns_current_source_and_moving_target_endpoint_segment():
    from orbit_wars_rl.core.geometry import predict_launch, trajectory_crosses_sun

    src = p(0, 0, 0)
    target = p(1, 45, 60)

    assert trajectory_crosses_sun((src.x, src.y), (target.x, target.y))
    launch = predict_launch(src, target, angular_velocity=0.05, fleet_speed=2.0)

    assert launch.source_xy == (src.x, src.y)
    assert not math.isclose(launch.target_xy[1], target.y)
    assert not trajectory_crosses_sun(launch.source_xy, launch.target_xy)


def test_predict_launch_sun_check_uses_current_source_not_predicted_source():
    from orbit_wars_rl.core.geometry import predict_launch, trajectory_crosses_sun

    src = p(0, 10, 30)
    target = p(1, 50, 45)

    launch = predict_launch(src, target, angular_velocity=0.05, fleet_speed=2.0)

    assert launch.source_xy == (src.x, src.y)
    assert not math.isclose(launch.target_xy[0], target.x)
    assert trajectory_crosses_sun(launch.source_xy, launch.target_xy)


def test_orbit_to_orbit_intercept_accounts_for_target_radius():
    src = Planet(0, 1, 45, 80, 5, 50, 1)
    target = Planet(1, 0, 70, 45, 6, 50, 1)
    angular_velocity = 0.07
    fleet_speed = 3.0

    launch = predict_launch(src, target, angular_velocity=angular_velocity, fleet_speed=fleet_speed)

    # Travel time from source edge to predicted target edge should match fleet travel.
    path_distance = distance_xy(src.x, src.y, launch.target_xy[0], launch.target_xy[1]) - src.radius - target.radius
    travel_time = path_distance / fleet_speed

    expected_target_xy = predicted_planet_position(target, travel_time, angular_velocity)

    assert math.isclose(launch.target_xy[0], expected_target_xy[0], rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(launch.target_xy[1], expected_target_xy[1], rel_tol=1e-4, abs_tol=1e-4)
