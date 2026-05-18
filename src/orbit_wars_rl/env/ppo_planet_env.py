"""Gymnasium planet-step wrapper for PPO Orbit Wars training."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - minimal non-RL installations.
    class _MiniArray(list):
        @property
        def shape(self):
            return (len(self),)

        def reshape(self, *shape):
            return self

        def tolist(self):
            return list(self)

    class _MiniNumpy:
        inf = float("inf")
        float32 = float

        @staticmethod
        def asarray(value, dtype=None):
            return _MiniArray(float(v) for v in value)

        @staticmethod
        def zeros(shape, dtype=None):
            return _MiniArray(0.0 for _ in range(shape[0]))

        @staticmethod
        def full(shape, value, dtype=None):
            return _MiniArray(float(value) for _ in range(shape[0]))

    np = _MiniNumpy()  # type: ignore[assignment]

try:  # Keep non-RL imports usable when gymnasium is not installed.
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - exercised only in minimal installations.
    class _Box:
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: Any):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

        def sample(self):
            low = 0.0 if self.low == -np.inf else self.low
            high = 1.0 if self.high == np.inf else self.high
            return np.full(self.shape, (low + high) / 2.0, dtype=self.dtype)

    class _Spaces:
        Box = _Box

    class _Env:
        pass

    class _Gym:
        Env = _Env

    gym = _Gym()  # type: ignore[assignment]
    spaces = _Spaces()  # type: ignore[assignment]

from orbit_wars_rl.agents.starter_agent import StarterAgent
from orbit_wars_rl.core.actions import ActionDecodeConfig, decode_model_outputs
from orbit_wars_rl.core.candidates import CandidateConfig, comet_ids_from_obs
from orbit_wars_rl.core.geometry import sun_collision_radius_from_obs
from orbit_wars_rl.core.observations import ObservationBuilder
from orbit_wars_rl.core.planets import total_production
from orbit_wars_rl.core.types import Planet, parse_planets, rows
from orbit_wars_rl.env.kaggle_env import require_kaggle_env
from orbit_wars_rl.training.reward import RewardShapingConfig, planet_capture_reward, ships_sent_reward


class _FakeOrbitWarsBackend:
    """Tiny deterministic backend for smoke tests only; it is not game physics."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.done = False
        self.turn = 0
        self.obs = self._initial_obs()

    def _initial_obs(self) -> dict[str, Any]:
        return {
            "player": 0,
            "angular_velocity": 0.03,
            "fleet_speed": 1.0,
            "comet_planet_ids": [],
            "comets": [],
            "fleets": [],
            "planets": [
                [0, 0, 80.0, 20.0, 2.0, 20, 3.0],
                [1, 0, 75.0, 25.0, 2.0, 15, 3.0],
                [2, 1, 20.0, 80.0, 2.0, 20, 3.0],
                [3, -1, 70.0, 18.0, 1.7, 8, 5.0],
                [4, -1, 62.0, 16.0, 1.5, 6, 4.0],
            ],
            "initial_planets": [],
            "step": 0,
        }

    def reset(self) -> dict[str, Any]:
        self.done = False
        self.turn = 0
        self.obs = self._initial_obs()
        return self.obs

    def observation(self, player: int) -> dict[str, Any]:
        return {**self.obs, "player": player}

    def step(self, actions: list[list[list[float | int]]]) -> dict[str, Any]:
        del actions
        self.turn += 1
        self.obs["step"] = self.turn
        # Deterministic production-control event: player 0 captures production on turn advance.
        for planet in self.obs["planets"]:
            if int(planet[0]) == 3 and self.turn == 1:
                planet[1] = 0
            if int(planet[1]) in (0, 1):
                planet[5] = int(planet[5]) + 1
        if self.turn >= 25:
            self.done = True
        return self.obs


def _as_observation_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Unsupported observation type: {type(value)!r}")


def _extract_player_observation(env: Any, player: int) -> dict[str, Any]:
    if isinstance(env, _FakeOrbitWarsBackend):
        return env.observation(player)
    state = getattr(env, "state", None)
    if state:
        item = state[player]
        obs = item.get("observation") if isinstance(item, dict) else getattr(item, "observation", item)
        return {**_as_observation_dict(obs), "player": player}
    if hasattr(env, "toJSON"):
        data = env.toJSON()
        steps = data.get("steps") if isinstance(data, dict) else None
        if steps:
            latest = steps[-1][player]
            obs = latest.get("observation", latest) if isinstance(latest, dict) else latest
            return {**_as_observation_dict(obs), "player": player}
    raise RuntimeError("Could not extract a player observation from the Kaggle environment state.")


def _step_kaggle_env(env: Any, actions_for_player0: list, actions_for_player1: list) -> None:
    if not callable(getattr(env, "step", None)):
        raise RuntimeError(
            "PPO planet-step training requires stepwise Kaggle environment access, but this "
            "orbit_wars environment does not expose env.step(...)."
        )
    try:
        env.step([actions_for_player0, actions_for_player1])
    except Exception as exc:
        raise RuntimeError(
            "PPO planet-step training requires direct env.step([actions0, actions1]) support. "
            "This Kaggle Orbit Wars installation did not accept stepwise actions."
        ) from exc


def _is_kaggle_done(env: Any) -> bool:
    if isinstance(env, _FakeOrbitWarsBackend):
        return env.done
    done = getattr(env, "done", None)
    if done is not None:
        return bool(done)
    state = getattr(env, "state", None)
    if state:
        statuses = [s.get("status") if isinstance(s, dict) else getattr(s, "status", None) for s in state]
        return any(str(status).upper() in {"DONE", "ERROR", "INVALID"} for status in statuses)
    return False


class OrbitWarsPlanetStepEnv(gym.Env):
    """Each SB3 step controls one owned planet; full Kaggle turn advances after all sources."""

    metadata = {"render_modes": []}
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
    action_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)

    def __init__(
        self,
        opponent: str = "starter",
        opponent_model: str | Path | None = None,
        candidate_player: int = 0,
        seed: int = 0,
        max_episode_turns: int = 400,
        require_kaggle: bool = True,
    ):
        if opponent == "starter":
            if opponent_model is not None:
                raise ValueError("opponent_model must be None when opponent='starter'")
        elif opponent == "model":
            if opponent_model is None:
                raise ValueError("opponent_model is required when opponent='model'")
        else:
            raise ValueError("opponent must be one of: 'starter', 'model'")
        if candidate_player not in (0, 1):
            raise ValueError("candidate_player must be 0 or 1")
        self.opponent = opponent
        self.opponent_model = opponent_model
        self.candidate_player = candidate_player
        self.seed_value = seed
        self.max_episode_turns = max_episode_turns
        self.require_kaggle = require_kaggle
        self.candidate_config = CandidateConfig()
        self.action_config = ActionDecodeConfig()
        self.builder = ObservationBuilder(self.candidate_config)
        self.reward_config = RewardShapingConfig()
        self.env: Any | None = None
        self.opponent_agent: Any | None = None
        self._opponent_policy: Any | None = None
        self.obs: dict[str, Any] = {}
        self.sources: list[Planet] = []
        self.current_source_index = 0
        self.buffered_actions: list = []
        self.previous_total_production = 0.0
        self.turn_index = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self.seed_value = seed
        self.env = require_kaggle_env(debug=True) if self.require_kaggle else _FakeOrbitWarsBackend(self.seed_value)
        self.opponent_agent = self._build_opponent_agent()
        reset = getattr(self.env, "reset", None)
        if callable(reset):
            reset()
        self.obs = _extract_player_observation(self.env, self.candidate_player)
        self.turn_index = 0
        self.buffered_actions = []
        self._rebuild_sources()
        self.previous_total_production = total_production(parse_planets(self.obs), self.candidate_player)
        return self._current_obs(), {"source_id": self._current_source_id(), "no_source": not self.sources}

    def step(self, action):
        if self.env is None:
            raise RuntimeError("reset() must be called before step().")
        if not self.sources:
            return self._advance_turn([])
        source = self.sources[self.current_source_index]
        planets = parse_planets(self.obs)
        comet_ids = comet_ids_from_obs(self.obs)
        _model_obs, chosen = self.builder.build_for_source(
            source,
            planets,
            self.candidate_player,
            previous_total_production=self.previous_total_production,
            comet_ids=comet_ids,
        )
        decoded = decode_model_outputs(
            source,
            chosen,
            np.asarray(action, dtype=np.float32).reshape(-1).tolist(),
            self.action_config,
            angular_velocity=float(self.obs.get("angular_velocity", 0.0)),
            fleet_speed=float(self.obs.get("fleet_speed", 1.0)),
            sun_radius=sun_collision_radius_from_obs(self.obs),
        )
        decoded_rows = rows(decoded)
        send_reward = ships_sent_reward(decoded_rows, self.reward_config)
        self.buffered_actions.extend(decoded_rows)
        self.current_source_index += 1
        if self.current_source_index < len(self.sources):
            return (
                self._current_obs(),
                send_reward,
                False,
                False,
                {
                    "turn_advanced": False,
                    "source_id": self._current_source_id(),
                    "send_reward": send_reward,
                },
            )
        return self._advance_turn(self.buffered_actions, send_reward)

    def _advance_turn(self, candidate_actions: list, send_reward: float = 0.0):
        assert self.env is not None
        opponent_player = 1 - self.candidate_player
        opponent_obs = _extract_player_observation(self.env, opponent_player)
        if self.opponent_agent is None:
            self.opponent_agent = self._build_opponent_agent()
        opponent_actions = self.opponent_agent.act(opponent_obs)
        actions0 = candidate_actions if self.candidate_player == 0 else opponent_actions
        actions1 = opponent_actions if self.candidate_player == 0 else candidate_actions
        production_before = self.previous_total_production
        planets_before = parse_planets(self.obs)
        _step_kaggle_env(self.env, actions0, actions1)
        self.turn_index += 1
        self.obs = _extract_player_observation(self.env, self.candidate_player)
        planets_after = parse_planets(self.obs)
        current_total = total_production(planets_after, self.candidate_player)
        production_delta = float(current_total - production_before)
        capture_reward = planet_capture_reward(
            planets_before, planets_after, self.candidate_player, self.reward_config
        )
        reward = float(production_delta + capture_reward + send_reward)
        self.previous_total_production = current_total
        buffered = list(candidate_actions)
        self.buffered_actions = []
        self._rebuild_sources()
        planets = planets_after
        candidate_has_planets = any(p.owner == self.candidate_player for p in planets)
        opponent_has_planets = any(p.owner == opponent_player for p in planets)
        terminated = _is_kaggle_done(self.env) or not candidate_has_planets or not opponent_has_planets
        truncated = self.turn_index >= self.max_episode_turns
        return (
            self._current_obs(),
            reward,
            bool(terminated),
            bool(truncated),
            {
                "turn_advanced": True,
                "production_before": production_before,
                "production_after": current_total,
                "production_delta": production_delta,
                "capture_reward": capture_reward,
                "send_reward": send_reward,
                "buffered_actions": buffered,
                "turn_index": self.turn_index,
                "source_id": self._current_source_id(),
            },
        )

    def _build_opponent_agent(self) -> Any:
        if self.opponent == "starter":
            return StarterAgent()
        if self._opponent_policy is None:
            from orbit_wars_rl.models.sb3_policy import SB3PolicyAdapter

            assert self.opponent_model is not None
            self._opponent_policy = SB3PolicyAdapter.load(self.opponent_model, device="cpu")
        from orbit_wars_rl.agents.model_agent import ModelAgent

        return ModelAgent(policy=self._opponent_policy)

    def _rebuild_sources(self) -> None:
        planets = parse_planets(self.obs)
        comet_ids = comet_ids_from_obs(self.obs)
        self.sources = [p for p in planets if p.owner == self.candidate_player and p.id not in comet_ids]
        self.current_source_index = 0

    def _current_source_id(self) -> int | None:
        if not self.sources:
            return None
        return self.sources[min(self.current_source_index, len(self.sources) - 1)].id

    def _current_obs(self) -> np.ndarray:
        if not self.sources:
            return np.zeros((15,), dtype=np.float32)
        source = self.sources[min(self.current_source_index, len(self.sources) - 1)]
        vec, _chosen = self.builder.build_for_source(
            source,
            parse_planets(self.obs),
            self.candidate_player,
            previous_total_production=self.previous_total_production,
            comet_ids=comet_ids_from_obs(self.obs),
        )
        return np.asarray(list(vec), dtype=np.float32)
