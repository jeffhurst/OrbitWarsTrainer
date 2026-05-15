import math
from orbit_wars_rl.agents.starter_agent import StarterAgent


def test_starter_sends_one_ship_to_each_selected_target_with_angles():
    obs = {
        "player": 0,
        "comet_planet_ids": [],
        "planets": [
            [0, 0, 80, 80, 10, 4, 1],
            [1, -1, 90, 80, 10, 5, 4],
            [2, -1, 80, 90, 10, 5, 3],
        ],
        "fleets": [],
    }
    actions = StarterAgent().act(obs)
    assert len(actions) == 2
    assert all(a[2] == 1 for a in actions)
    assert math.isclose(actions[0][1], 0.0)
    assert math.isclose(actions[1][1], math.pi / 2)


def test_starter_skips_when_not_enough_ships():
    obs = {
        "player": 0,
        "comet_planet_ids": [],
        "planets": [
            [0, 0, 80, 80, 10, 1, 1],
            [1, -1, 90, 80, 10, 5, 4],
            [2, -1, 80, 90, 10, 5, 3],
        ],
    }
    assert StarterAgent().act(obs) == []
