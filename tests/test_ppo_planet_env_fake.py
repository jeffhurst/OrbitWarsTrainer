import pytest

from orbit_wars_rl.env.ppo_planet_env import OrbitWarsPlanetStepEnv, _FakeOrbitWarsBackend


def test_fake_planet_step_env_shapes_and_production_delta_reward():
    env = OrbitWarsPlanetStepEnv(require_kaggle=False, max_episode_turns=4)
    obs, info = env.reset(seed=123)
    assert obs.shape == (15,)
    assert info["source_id"] == 0
    assert env.action_space.shape == (4,)

    next_obs, reward, terminated, truncated, info = env.step([0] * 4)
    assert next_obs.shape == (15,)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["turn_advanced"] is False

    next_obs, reward, terminated, truncated, info = env.step([0] * 4)
    assert next_obs.shape == (15,)
    assert info["turn_advanced"] is True
    assert info["production_delta"] == info["production_after"] - info["production_before"]
    assert info["capture_reward"] == pytest.approx(90.0)
    assert reward == pytest.approx(info["reward_total"])


def test_fake_planet_step_env_loads_model_opponent(tmp_path):
    from orbit_wars_rl.agents.model_agent import ModelAgent
    from orbit_wars_rl.models.policy import NumpyPolicy

    model_path = tmp_path / "policy.json"
    saved_policy = NumpyPolicy.random(1)
    saved_policy.save(model_path)

    env = OrbitWarsPlanetStepEnv(opponent="model", opponent_model=model_path, require_kaggle=False)
    env.reset(seed=123)

    assert isinstance(env.opponent_agent, ModelAgent)
    assert env.opponent_agent.policy.predict([0.0] * 15) == saved_policy.predict([0.0] * 15)


def test_fake_planet_step_env_wraps_loaded_policy_from_any_supported_artifact(monkeypatch, tmp_path):
    from orbit_wars_rl.agents.model_agent import ModelAgent
    from orbit_wars_rl.env import ppo_planet_env

    class LoadedPolicy:
        def predict(self, obs):
            return [0] * 4

    loaded_policy = LoadedPolicy()
    model_path = tmp_path / "ppo_orbit_wars.zip"
    model_path.write_text("placeholder")
    calls = []

    def fake_load_any_policy(path):
        calls.append(path)
        return loaded_policy

    monkeypatch.setattr(ppo_planet_env, "load_any_policy", fake_load_any_policy)

    env = OrbitWarsPlanetStepEnv(opponent="model", opponent_model=model_path, require_kaggle=False)
    env.reset(seed=123)

    assert calls == [model_path]
    assert isinstance(env.opponent_agent, ModelAgent)
    assert env.opponent_agent.policy is loaded_policy


def test_fake_planet_step_env_requires_reset_opponent_agent_before_advancing():
    env = OrbitWarsPlanetStepEnv(require_kaggle=False)
    env.env = _FakeOrbitWarsBackend(0)

    with pytest.raises(RuntimeError, match="opponent agent"):
        env._advance_turn([])


def test_fake_planet_step_env_adds_early_win_terminal_bonus(monkeypatch):
    from orbit_wars_rl.env import ppo_planet_env

    def fake_step_kaggle_env(backend, actions_for_player0, actions_for_player1):
        del actions_for_player0, actions_for_player1
        backend.turn += 1
        backend.obs["step"] = backend.turn
        for planet in backend.obs["planets"]:
            if int(planet[0]) == 2:
                planet[1] = 0
        backend.done = True

    monkeypatch.setattr(ppo_planet_env, "_step_kaggle_env", fake_step_kaggle_env)
    env = OrbitWarsPlanetStepEnv(require_kaggle=False, max_episode_turns=4)
    env.reset(seed=123)

    env.step([0] * 4)
    _next_obs, reward, terminated, truncated, info = env.step([0] * 4)

    assert terminated is True
    assert truncated is False
    assert info["terminal_reward"] == pytest.approx(240.0 + 80.0 * (3 / 4))
    assert reward == pytest.approx(info["reward_total"])


def test_fake_planet_step_env_adds_ship_score_terminal_bonus_on_truncation(monkeypatch):
    from orbit_wars_rl.env import ppo_planet_env

    def fake_step_kaggle_env(backend, actions_for_player0, actions_for_player1):
        del actions_for_player0, actions_for_player1
        backend.turn += 1
        backend.obs["step"] = backend.turn
        backend.obs["planets"] = [
            [0, 0, 0.0, 0.0, 1.0, 100, 1.0],
            [1, 1, 0.0, 0.0, 1.0, 1, 50.0],
        ]

    monkeypatch.setattr(ppo_planet_env, "_step_kaggle_env", fake_step_kaggle_env)
    env = OrbitWarsPlanetStepEnv(require_kaggle=False, max_episode_turns=1)
    env.reset(seed=123)

    _next_obs, reward, terminated, truncated, info = env._advance_turn([])

    assert terminated is False
    assert truncated is True
    assert -100.0 <= info["terminal_reward"] <= 100.0
    assert reward == pytest.approx(info["reward_total"])

def test_reset_uses_new_random_map_seed_for_each_game(monkeypatch):
    from orbit_wars_rl.env import ppo_planet_env

    class DummyKaggleEnv:
        def reset(self):
            return None

    calls = []

    def fake_require_kaggle_env(**kwargs):
        calls.append(kwargs)
        return DummyKaggleEnv()

    def fake_extract_player_observation(_env, player):
        return _FakeOrbitWarsBackend(0).observation(player)

    monkeypatch.setattr(ppo_planet_env, "require_kaggle_env", fake_require_kaggle_env)
    monkeypatch.setattr(ppo_planet_env, "_extract_player_observation", fake_extract_player_observation)

    env = OrbitWarsPlanetStepEnv(require_kaggle=True, seed=7)
    env.reset()
    env.reset()

    assert len(calls) == 2
    assert calls[0]["debug"] is True
    assert calls[1]["debug"] is True
    assert calls[0]["configuration"]["randomSeed"] != calls[1]["configuration"]["randomSeed"]
    assert calls[0]["configuration"]["randomSeed"] in ppo_planet_env.MAP_SEEDS
    assert calls[1]["configuration"]["randomSeed"] in ppo_planet_env.MAP_SEEDS
