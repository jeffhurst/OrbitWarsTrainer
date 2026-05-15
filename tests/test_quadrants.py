from orbit_wars_rl.core.geometry import counterclockwise_quadrant, quadrant_of


def test_quadrants_screen_coordinates():
    assert quadrant_of(50, 49.9) == 1
    assert quadrant_of(49.9, 49.9) == 2
    assert quadrant_of(49.9, 50) == 3
    assert quadrant_of(50, 50) == 4


def test_counterclockwise_mapping_screen_coordinates():
    assert [counterclockwise_quadrant(q) for q in [1, 2, 3, 4]] == [2, 3, 4, 1]
