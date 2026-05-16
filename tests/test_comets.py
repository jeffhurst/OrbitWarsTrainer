from orbit_wars_rl.core.candidates import select_candidates
from orbit_wars_rl.core.comets import CometController, closest_non_comet_planet
from orbit_wars_rl.core.types import Planet


def planet(id, owner, x, y, ships=5, prod=1, radius=10):
    return Planet(id, owner, x, y, radius, ships, prod)


def test_comets_excluded_from_normal_candidate_selection():
    source = planet(0, 0, 80, 80)
    comet = planet(1, -1, 81, 80, prod=5)
    normal = planet(2, -1, 82, 80, prod=1)
    assert [
        p.id for p in select_candidates(source, [source, comet, normal], 0, {1}).candidates
    ] == [2]


def test_closest_non_comet_planet_from_comet():
    comet = planet(9, 0, 50, 50)
    near = planet(1, -1, 55, 50)
    far = planet(2, -1, 70, 50)
    assert closest_non_comet_planet(comet, [comet, far, near], {9}).id == 1


def test_forced_comet_launch_generated_after_trigger():
    controller = CometController(reserve_ships=1)
    obs1 = {
        "player": 0,
        "comet_planet_ids": [9],
        "planets": [[9, -1, 80, 80, 1, 5, 1], [1, -1, 85, 80, 1, 5, 1]],
    }
    assert controller.update_and_forced_actions(obs1, 0) == []
    obs2 = {
        "player": 0,
        "comet_planet_ids": [9],
        "planets": [[9, 0, 80, 80, 1, 8, 1], [1, -1, 85, 80, 1, 5, 1]],
    }
    assert controller.update_and_forced_actions(obs2, 0) == []
    actions = controller.update_and_forced_actions(obs2, 0)
    assert len(actions) == 1
    assert actions[0].from_planet_id == 9
    assert actions[0].num_ships == 7


def test_forced_comet_launch_skips_sun_crossing_path():
    controller = CometController(reserve_ships=1)
    obs1 = {
        "player": 0,
        "comet_planet_ids": [9],
        "planets": [[9, -1, 40, 50, 1, 5, 1], [1, -1, 60, 50, 1, 5, 1]],
        "sun_radius": 5,
    }
    assert controller.update_and_forced_actions(obs1, 0) == []
    obs2 = {
        "player": 0,
        "comet_planet_ids": [9],
        "planets": [[9, 0, 40, 50, 1, 8, 1], [1, -1, 60, 50, 1, 5, 1]],
        "sun_radius": 5,
    }
    assert controller.update_and_forced_actions(obs2, 0) == []
    assert controller.update_and_forced_actions(obs2, 0) == []
