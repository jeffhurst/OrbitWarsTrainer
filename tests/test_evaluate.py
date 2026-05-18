import json

import pytest

from orbit_wars_rl.evaluation import evaluate


class ConstantPolicy:
    def predict(self, obs):
        return [0.0] * 9


class ScriptedPlanetStepEnv:
    instances = []

    def __init__(self, *, require_kaggle, seed, max_episode_turns):
        self.require_kaggle = require_kaggle
        self.seed = seed
        self.max_episode_turns = max_episode_turns
        self.obs = {
            "player": 0,
            "fleets": [],
            "planets": [
                [0, 0, 0.0, 0.0, 1.0, 10, 1.0],
                [1, 1, 0.0, 0.0, 1.0, 10, 1.0],
            ],
        }
        self.steps = 0
        self.__class__.instances.append(self)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        return [0.0] * 15, {"source_id": 0}

    def step(self, action):
        del action
        self.steps += 1
        self.obs = {
            "player": 0,
            "fleets": [],
            "planets": [
                [0, 0, 0.0, 0.0, 1.0, 100, 1.0],
                [1, 1, 0.0, 0.0, 1.0, 1, 50.0],
            ],
        }
        return [0.0] * 15, 123.0, False, True, {"turn_advanced": True, "production_delta": 7.0}


@pytest.mark.parametrize(
    ("runner", "expected_backend", "expected_require_kaggle"),
    [
        (evaluate._evaluate_fake, "fake-smoke", False),
        (evaluate._evaluate_kaggle, "kaggle", True),
    ],
)
def test_evaluation_metrics_use_shaped_reward_and_ship_score_win_rate(
    monkeypatch, tmp_path, runner, expected_backend, expected_require_kaggle
):
    ScriptedPlanetStepEnv.instances = []
    monkeypatch.setattr(evaluate, "OrbitWarsPlanetStepEnv", ScriptedPlanetStepEnv)

    path = runner(ConstantPolicy(), 1, tmp_path)

    metrics = json.loads(path.read_text())
    assert metrics["backend"] == expected_backend
    assert metrics["average_reward"] == pytest.approx(123.0)
    assert metrics["average_production_delta"] == pytest.approx(7.0)
    assert metrics["final_average_candidate_production"] < metrics["final_average_opponent_production"]
    assert metrics["final_average_candidate_score"] > metrics["final_average_opponent_score"]
    assert metrics["win_rate"] == pytest.approx(1.0)
    assert ScriptedPlanetStepEnv.instances[0].require_kaggle is expected_require_kaggle
