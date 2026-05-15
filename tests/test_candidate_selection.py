from orbit_wars_rl.core.candidates import CandidateConfig, select_candidates
from orbit_wars_rl.core.types import Planet


def planet(id, owner, x, y, ships=5, prod=1, radius=1):
    return Planet(id, owner, x, y, radius, ships, prod)


def test_static_radius_orbiting_quadrant_comets_and_source_excluded():
    source = planet(0, 0, 80, 80, prod=1)  # Q4, counterclockwise Q1
    static_in = planet(1, -1, 92, 80, prod=3, radius=10)  # non-orbiting, within 25
    static_out = planet(2, -1, 20, 80, prod=5, radius=10)  # non-orbiting, outside 25
    orbit_good = planet(3, -1, 60, 40, prod=4)  # orbiting Q1
    orbit_bad = planet(4, -1, 40, 40, prod=5)  # orbiting Q2
    comet = planet(5, -1, 82, 82, prod=9)
    selected = select_candidates(source, [source, static_in, static_out, orbit_good, orbit_bad, comet], 0, {5}).candidates
    assert [p.id for p in selected] == [3, 1]


def test_top_4_sorted_by_production_owner_priority_then_distance():
    source = planet(0, 0, 80, 80, prod=1)
    planets = [
        source,
        planet(1, 0, 81, 80, prod=5, radius=10),
        planet(2, -1, 82, 80, prod=5, radius=10),
        planet(3, 1, 83, 80, prod=5, radius=10),
        planet(4, -1, 84, 80, prod=4, radius=10),
        planet(5, -1, 85, 80, prod=3, radius=10),
    ]
    selected = select_candidates(source, planets, 0, set()).candidates
    assert [p.id for p in selected] == [3, 2, 1, 4]


def test_distance_tiebreak_with_same_owner_priority():
    source = planet(0, 0, 80, 80)
    near = planet(1, -1, 82, 80, prod=2, radius=10)
    far = planet(2, -1, 90, 80, prod=2, radius=10)
    assert [p.id for p in select_candidates(source, [source, far, near], 0, set()).candidates] == [1, 2]
