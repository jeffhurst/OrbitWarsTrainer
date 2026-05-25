import pytest

from orbit_wars_rl.core.candidates import CandidateConfig
from orbit_wars_rl.core.observations import ObservationBuilder
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


def test_planet_step_env_observes_and_decodes_filtered_candidates():
    env = OrbitWarsPlanetStepEnv(require_kaggle=False, max_episode_turns=4)
    env.reset(seed=123)
    env.candidate_config = CandidateConfig(static_radius=120)
    env.builder = ObservationBuilder(env.candidate_config)
    env.obs = {
        "player": 0,
        "planets": [
            [0, 0, 0, 50, 5, 10, 1],
            [1, 1, 100, 50, 5, 1, 9],
            [2, 1, 0, 80, 5, 2, 6],
        ],
        "fleets": [],
    }
    env.previous_total_production = 1.0
    env._rebuild_sources()

    current_obs = env._current_obs()
    _next_obs, _reward, _terminated, _truncated, info = env.step([5, 0, 0, 0])

    assert current_obs[3:6].tolist() == [-1, -2, 6]
    assert len(info["buffered_actions"]) == 1


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
    assert info["terminal_reward"] == pytest.approx(600.0 + 300.0 * (3 / 4))
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
    assert calls[0]["configuration"]["seed"] != calls[1]["configuration"]["seed"]
    assert calls[0]["configuration"]["seed"] in ppo_planet_env.MAP_SEEDS
    assert calls[1]["configuration"]["seed"] in ppo_planet_env.MAP_SEEDS


def test_episode_components_have_no_pressure_or_waste_and_reward_totals_match():
    env = OrbitWarsPlanetStepEnv(require_kaggle=False, max_episode_turns=1)
    _obs, reset_info = env.reset(seed=123)
    _next_obs, reward, terminated, truncated, info = env._advance_turn([])
    assert terminated or truncated
    components = info["episode_components"]
    assert "reward/pressure" not in components
    assert "reward/waste_penalty" not in components
    assert "game/map_seed" in components
    assert components["reward/total"] == pytest.approx(reward)
    assert reset_info["map_seed"] == int(components["game/map_seed"])


def test_source_tactical_reward_rewards_saving_and_penalizes_missed_attacks():
    from orbit_wars_rl.core.types import Planet

    env = OrbitWarsPlanetStepEnv(require_kaggle=False)
    source = Planet(0, 0, 80, 80, 2.0, 10, 1.0)
    weak_enemy = Planet(2, 1, 83, 80, 2.0, 9, 1.0)

    assert env._source_tactical_reward(source, [weak_enemy], [], []) > 0.0

    under_send_reward = env._source_tactical_reward(
        source,
        [weak_enemy],
        [[0, 0.0, 1]],
        [1.0, 0.1] + [0.0] * 6,
    )
    assert under_send_reward < 0.0

    no_attack_reward = env._source_tactical_reward(source, [weak_enemy], [[0, 0.0, 10]], [0.0] * 8)
    assert no_attack_reward == pytest.approx(0.0)


def test_source_tactical_reward_two_value_branch_uses_decoder_min_send_floor():
    from orbit_wars_rl.core.types import Planet

    env = OrbitWarsPlanetStepEnv(require_kaggle=False)
    source = Planet(0, 0, 80, 80, 2.0, 20, 1.0)
    enemy = Planet(1, 1, 82, 80, 2.0, 8, 1.0)

    reward = env._source_tactical_reward(source, [enemy], [[0, 0.0, 0]], [1, 0.0])

    assert reward == pytest.approx(0.0)
