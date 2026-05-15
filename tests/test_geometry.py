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
