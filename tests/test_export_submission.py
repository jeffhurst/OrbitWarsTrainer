import math
import runpy
import subprocess
import sys


def standalone_namespace():
    namespace = {}
    exec(runpy.run_path("scripts/export_submission.py")["STANDALONE"], namespace)
    return namespace


def generated_submission_namespace(tmp_path):
    out = tmp_path / "submission.py"
    subprocess.run(
        [sys.executable, "scripts/export_submission.py", "--out", str(out)],
        check=True,
        text=True,
        capture_output=True,
    )
    return runpy.run_path(out)


def test_standalone_comet_ids_supports_all_observation_shapes():
    ns = standalone_namespace()

    assert ns["_comet_ids"](
        {
            "comet_planet_ids": [1],
            "comets": [
                {"planet_ids": [2]},
                ([3], "ignored"),
            ],
        }
    ) == {1, 2, 3}


def test_standalone_agent_excludes_comets_declared_in_comets_field():
    ns = standalone_namespace()
    obs = {
        "player": 0,
        "comets": [{"planet_ids": [1]}],
        "planets": [
            [0, 0, 80, 80, 10, 5, 1],
            [1, -1, 81, 80, 10, 5, 9],
            [2, -1, 82, 80, 10, 5, 1],
        ],
    }

    assert len(ns["agent"](obs)) == 1


def test_generated_submission_keeps_static_targets_on_direct_atan2_path(tmp_path):
    ns = generated_submission_namespace(tmp_path)
    obs = {
        "player": 0,
        "angular_velocity": 0.03,
        "planets": [
            [0, 0, 80, 80, 10, 5, 1],
            [1, -1, 90, 80, 10, 5, 9],
        ],
    }

    actions = ns["agent"](obs)

    assert actions == [[0, math.atan2(80 - 80, 90 - 80), 1]]


def test_generated_submission_leads_orbiting_targets_but_not_when_velocity_is_zero(tmp_path):
    ns = generated_submission_namespace(tmp_path)
    base_obs = {
        "player": 0,
        "planets": [
            [0, 0, 80, 80, 10, 5, 1],
            [1, -1, 60, 40, 1, 5, 9],
        ],
    }
    direct_angle = math.atan2(40 - 80, 60 - 80)

    static_orbit_actions = ns["agent"]({**base_obs, "angular_velocity": 0.0})
    moving_orbit_actions = ns["agent"]({**base_obs, "angular_velocity": 0.03})

    assert static_orbit_actions == [[0, direct_angle, 1]]
    assert len(moving_orbit_actions) == 1
    assert moving_orbit_actions[0][0] == 0
    assert moving_orbit_actions[0][2] == 1
    assert not math.isclose(moving_orbit_actions[0][1], direct_angle)
    assert math.isclose(
        moving_orbit_actions[0][1],
        ns["_intercept_angle"](base_obs["planets"][0], base_obs["planets"][1], 0.03),
    )


def test_generated_submission_skips_sun_crossing_starter_action(tmp_path):
    ns = generated_submission_namespace(tmp_path)
    obs = {
        "player": 0,
        "sun_radius": 5,
        "planets": [
            [0, 0, 40, 50, 1, 5, 1],
            [1, -1, 60, 50, 1, 5, 9],
        ],
    }

    assert ns["agent"](obs) == []


def test_generated_submission_allows_path_outside_sun_radius(tmp_path):
    ns = generated_submission_namespace(tmp_path)
    obs = {
        "player": 0,
        "sun_radius": 5,
        "planets": [
            [0, 0, 40, 56, 1, 5, 1],
            [1, -1, 60, 56, 1, 5, 9],
        ],
    }

    assert len(ns["agent"](obs)) == 1
