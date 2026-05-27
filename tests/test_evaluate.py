import json

import pytest

from orbit_wars_rl.evaluation import evaluate


class ConstantPolicy:
    def __init__(self):
        self.deterministic_values = []

    def predict(self, obs, deterministic=False):
        del obs
        self.deterministic_values.append(deterministic)
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

    policy = ConstantPolicy()
    path = runner(policy, 1, tmp_path)

    metrics = json.loads(path.read_text())
    assert metrics["backend"] == expected_backend
    assert metrics["average_reward"] == pytest.approx(123.0)
    assert metrics["average_production_delta"] == pytest.approx(7.0)
    assert metrics["final_average_candidate_production"] < metrics["final_average_opponent_production"]
    assert metrics["final_average_candidate_score"] > metrics["final_average_opponent_score"]
    assert metrics["win_rate"] == pytest.approx(1.0)
    assert ScriptedPlanetStepEnv.instances[0].require_kaggle is expected_require_kaggle
    assert policy.deterministic_values == [True]


class ScriptedMapSeedEnv:
    instances = []

    def __init__(
        self,
        *,
        opponent,
        opponent_model,
        candidate_player,
        require_kaggle,
        seed,
        max_episode_turns,
    ):
        self.opponent = opponent
        self.opponent_model = opponent_model
        self.candidate_player = candidate_player
        self.require_kaggle = require_kaggle
        self.seed = seed
        self.max_episode_turns = max_episode_turns
        self.map_seed = None
        self.__class__.instances.append(self)

    def reset(self, *, options=None):
        self.map_seed = int(options["map_seed"])
        return [0.0] * 15, {}

    def step(self, action):
        del action
        reward = 10.0 if self.map_seed == 11 else -5.0
        return [0.0] * 15, reward, True, False, {
            "turn_advanced": True,
            "turn_index": self.map_seed,
            "terminal_reward": reward,
        }


class DeterministicRecordingPolicy:
    def __init__(self):
        self.deterministic_values = []

    def predict(self, obs, deterministic=False):
        del obs
        self.deterministic_values.append(deterministic)
        return [0, 0, 0, 0], None


def test_map_seed_eval_uses_deterministic_predict_and_logs_per_seed(monkeypatch):
    ScriptedMapSeedEnv.instances = []
    monkeypatch.setattr(evaluate, "OrbitWarsPlanetStepEnv", ScriptedMapSeedEnv)
    policy = DeterministicRecordingPolicy()

    metrics, results = evaluate.evaluate_map_seeds_deterministic(
        policy,
        [11, 12],
        require_kaggle=False,
        opponent="greedy",
        candidate_player=1,
        max_episode_turns=22,
    )

    assert policy.deterministic_values == [True, True]
    assert [result["map_seed"] for result in results] == [11.0, 12.0]
    assert metrics["eval/map_seed_1"] == pytest.approx(11.0)
    assert metrics["eval/map_seed_11/win_rate_deterministic"] == pytest.approx(1.0)
    assert metrics["eval/map_seed_12/win_rate_deterministic"] == pytest.approx(0.0)
    assert metrics["eval/map_seed/win_rate_deterministic"] == pytest.approx(0.5)
    assert "eval/map_seed_11/noop_rate" in metrics
    assert "eval/map_seed_11/pass_rate" in metrics
    assert "eval/map_seed_11/ships_sent_mean" in metrics
    assert "eval/map_seed_11/target_choice_count_0" in metrics
    assert ScriptedMapSeedEnv.instances[0].opponent == "greedy"
    assert ScriptedMapSeedEnv.instances[0].candidate_player == 1
    assert ScriptedMapSeedEnv.instances[0].max_episode_turns == 22
