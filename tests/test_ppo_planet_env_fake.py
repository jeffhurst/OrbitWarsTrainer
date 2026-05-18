import pytest

from orbit_wars_rl.env.ppo_planet_env import OrbitWarsPlanetStepEnv, _FakeOrbitWarsBackend


def test_fake_planet_step_env_shapes_and_production_delta_reward():
    env = OrbitWarsPlanetStepEnv(require_kaggle=False, max_episode_turns=4)
    obs, info = env.reset(seed=123)
    assert obs.shape == (15,)
    assert info["source_id"] == 0
    assert env.action_space.shape == (9,)

    next_obs, reward, terminated, truncated, info = env.step([0.0] * 9)
    assert next_obs.shape == (15,)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["turn_advanced"] is False

    next_obs, reward, terminated, truncated, info = env.step([0.0] * 9)
    assert next_obs.shape == (15,)
    assert info["turn_advanced"] is True
    assert info["production_delta"] == info["production_after"] - info["production_before"]
    assert info["capture_reward"] == pytest.approx(0.5)
    assert reward == pytest.approx(info["production_delta"] + info["capture_reward"] + info["send_reward"])


def test_fake_planet_step_env_loads_model_opponent(tmp_path):
    from orbit_wars_rl.agents.model_agent import ModelAgent
    from orbit_wars_rl.models.policy import NumpyPolicy

    model_path = tmp_path / "policy.json"
    NumpyPolicy.random(1).save(model_path)

    env = OrbitWarsPlanetStepEnv(opponent="model", opponent_model=model_path, require_kaggle=False)
    env.reset(seed=123)

    assert isinstance(env.opponent_agent, ModelAgent)


def test_fake_planet_step_env_requires_reset_opponent_agent_before_advancing():
    env = OrbitWarsPlanetStepEnv(require_kaggle=False)
    env.env = _FakeOrbitWarsBackend(0)

    with pytest.raises(RuntimeError, match="opponent agent"):
        env._advance_turn([])
