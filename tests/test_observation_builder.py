from orbit_wars_rl.core.observations import ObservationBuilder
from orbit_wars_rl.core.candidates import CandidateConfig
from orbit_wars_rl.core.types import Planet


def planet(id, owner, x, y, ships, prod, radius=10):
    return Planet(id, owner, x, y, radius, ships, prod)


def test_observation_length_encoding_ship_signs_totals_and_delta():
    source = planet(0, 0, 80, 80, 10, 2)
    enemy = planet(1, 1, 82, 80, 7, 5)
    neutral = planet(2, -1, 83, 80, 6, 4)
    friendly = planet(3, 0, 84, 80, 5, 3)
    builder = ObservationBuilder()
    obs, chosen = builder.build_for_source(source, [source, enemy, neutral, friendly], 0, previous_total_production=3)
    assert obs.shape == (15,)
    assert obs[0] == 5  # source + friendly production
    assert obs[1] == 2
    assert obs[2] == 10
    assert [p.id for p in chosen] == [1, 2, 3]
    assert obs[3:6].tolist() == [-1, -7, 5]
    assert obs[6:9].tolist() == [0, -6, 4]
    assert obs[9:12].tolist() == [1, 5, 3]
    assert obs[12:15].tolist() == [0, 0, 0]


def test_filtered_observation_encodes_only_valid_trajectory_candidates():
    source = planet(0, 0, 0, 50, 10, 1, radius=5)
    sun_crossing = planet(1, 1, 100, 50, 1, 9, radius=5)
    valid = planet(2, 1, 0, 80, 2, 6, radius=5)
    builder = ObservationBuilder(CandidateConfig(static_radius=120))

    obs, chosen, launches = builder.build_filtered_for_source(
        source,
        [source, sun_crossing, valid],
        0,
        previous_total_production=1,
    )

    assert [p.id for p in chosen] == [2]
    assert len(launches) == 1
    assert obs[3:6].tolist() == [-1, -2, 6]
