import runpy


def standalone_namespace():
    namespace = {}
    exec(runpy.run_path("scripts/export_submission.py")["STANDALONE"], namespace)
    return namespace


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
