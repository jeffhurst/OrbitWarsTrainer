import json
import math
import runpy
import subprocess
import sys

import pytest


def standalone_namespace():
    namespace = {}
    exec(runpy.run_path("scripts/export_submission.py")["STANDALONE"], namespace)
    return namespace


def standalone_namespace_for_model(model):
    template = runpy.run_path("scripts/export_submission.py")["STANDALONE_TEMPLATE"]
    namespace = {}
    exec(template.replace("__MODEL__", json.dumps(model, separators=(",", ":"))), namespace)
    return namespace


def generated_submission_namespace(tmp_path, model=None):
    out = tmp_path / "submission.py"
    command = [sys.executable, "scripts/export_submission.py", "--out", str(out)]
    if model is not None:
        command.extend(["--model", str(model)])
    subprocess.run(
        command,
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


def test_standalone_candidate_selection_matches_orbiting_source_rules():
    ns = standalone_namespace()
    source = [0, 0, 90, 50, 1, 5, 1]
    planets = [
        source,
        [1, -1, 99, 99, 1, 4, 2],
        [2, -1, 99, 49, 10, 1, 9],
        [3, -1, 60, 40, 1, 3, 1],
        [4, -1, 90, 90, 1, 2, 1],
    ]

    selected = ns["_select_candidates"](source, planets, 0, set())

    assert [p[0] for p in selected] == [1, 4, 3]


def test_standalone_filtered_observation_removes_sun_crossing_candidate():
    ns = standalone_namespace()
    source = [0, 0, 40, 50, 5, 10, 1]
    crossing = [1, 1, 60, 50, 5, 1, 9]
    valid = [2, 1, 40, 60, 5, 2, 6]

    obs, chosen, launches = ns["_build_filtered_for_source"](
        source,
        [source, crossing, valid],
        0,
        1,
        set(),
        0.0,
        1.0,
        5.0,
    )

    assert [p[0] for p in chosen] == [2]
    assert len(launches) == 1
    assert obs[3:6] == [-1.0, -2.0, 6.0]


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
        ns["_predict_launch"](base_obs["planets"][0], base_obs["planets"][1], 0.03, 1.0)[0],
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


def test_generated_submission_ignores_legacy_ninth_noop_output(tmp_path):
    ns = generated_submission_namespace(tmp_path)
    source = [0, 0, 80, 80, 1, 10, 1]
    target = [1, -1, 90, 80, 1, 1, 1]

    actions = ns["_decode"](
        source,
        [target],
        [0.51, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        0.0,
        1.0,
        10.0,
    )

    assert len(actions) == 1


def test_standalone_multidiscrete_legacy_prediction_and_decode():
    bias = [0.0] * 24
    bias[5] = 10.0
    model = {
        "kind": "multidiscrete",
        "hidden": [],
        "action": {"w": [[0.0] * 24 for _ in range(15)], "b": bias},
        "nvec": [6, 6, 6, 6],
        "output_size": 4,
    }
    ns = standalone_namespace_for_model(model)

    prediction = ns["_predict"]([0.0] * 15)
    actions = ns["_decode"](
        [0, 0, 80, 80, 1, 21, 1],
        [[1, -1, 90, 80, 1, 1, 1], [2, -1, 80, 90, 1, 1, 1]],
        prediction,
        0.0,
        1.0,
        10.0,
    )

    assert prediction == [5.0, 0.0, 0.0, 0.0]
    assert len(actions) == 1
    assert actions[0][2] == 20


def test_export_numpy_policy_even_with_zip_suffix(tmp_path):
    from orbit_wars_rl.models.policy import NumpyPolicy

    policy = NumpyPolicy.random(3, hidden=4)
    model_path = tmp_path / "bootstrap_policy.zip"
    policy.save(model_path)

    ns = generated_submission_namespace(tmp_path, model=model_path)

    obs = [0.25] * 15
    assert ns["_MODEL"]["kind"] == "numpy"
    assert ns["_predict"](obs) == pytest.approx(policy.predict(obs))
