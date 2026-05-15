from orbit_wars_rl.core.observations import ObservationBuilder
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
    # sorted prod desc: enemy, neutral, friendly
    assert [p.id for p in chosen] == [1, 2, 3]
    assert obs[3:6].tolist() == [-1, -7, 5]
    assert obs[6:9].tolist() == [0, -6, 4]
    assert obs[9:12].tolist() == [1, 5, 3]
    assert obs[12:15].tolist() == [0, 0, 0]
