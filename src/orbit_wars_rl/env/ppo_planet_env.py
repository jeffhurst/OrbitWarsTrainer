"""Gymnasium planet-step wrapper for PPO Orbit Wars training."""
from __future__ import annotations

from pathlib import Path
import random
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

    class _MultiDiscrete:
        def __init__(self, nvec):
            self.nvec = nvec
            self.shape = (len(nvec),)

    class _Spaces:
        Box = _Box
        MultiDiscrete = _MultiDiscrete

    class _Env:
        pass

    class _Gym:
        Env = _Env

    gym = _Gym()  # type: ignore[assignment]
    spaces = _Spaces()  # type: ignore[assignment]

from orbit_wars_rl.agents.model_agent import ModelAgent
from orbit_wars_rl.agents.starter_agent import StarterAgent
from orbit_wars_rl.core.actions import ActionDecodeConfig, decode_model_outputs
from orbit_wars_rl.core.candidates import CandidateConfig, comet_ids_from_obs
from orbit_wars_rl.core.geometry import sun_collision_radius_from_obs
from orbit_wars_rl.core.observations import ObservationBuilder
from orbit_wars_rl.core.planets import total_production
from orbit_wars_rl.core.types import Planet, parse_planets, rows
from orbit_wars_rl.env.kaggle_env import require_kaggle_env
from orbit_wars_rl.models.save_load import load_any_policy
from orbit_wars_rl.training.reward import (
    RewardShapingConfig,
    game_outcome_reward,
    fleet_pressure_advantage,
    idle_ship_penalty,
    planet_capture_reward,
    player_score,
    production_advantage,
    score_advantage,
    action_targeting_reward,
    ships_sent_reward,
    strategic_score,
    timeout_outcome_reward,
)


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
    action_space = spaces.MultiDiscrete([6, 6, 6, 6])

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
        self._map_seed_rng = random.Random(seed)
        self._episode_index = 0
        self._last_map_seed: int | None = None
        self.max_episode_turns = max_episode_turns
        self.require_kaggle = require_kaggle
        self.candidate_config = CandidateConfig()
        self.action_config = ActionDecodeConfig()
        self.builder = ObservationBuilder(self.candidate_config)
        self.reward_config = RewardShapingConfig()
        self.env: Any | None = None
        self.opponent_agent: Any | None = None
        self.obs: dict[str, Any] = {}
        self.sources: list[Planet] = []
        self.current_source_index = 0
        self.buffered_actions: list = []
        self.previous_total_production = 0.0
        self.turn_index = 0
        self.episode_action_count = 0
        self.episode_invalid_action_count = 0
        self.episode_ships_sent_total = 0.0
        self.episode_target_choice_count = 0
        self.episode_enemy_target_count = 0
        self.episode_neutral_target_count = 0
        self.episode_self_target_count = 0
        self.episode_turn_count = 0
        self.episode_enemy_captures = 0
        self.episode_neutral_captures = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        options = options or {}
        reset_seed = seed if seed is not None else options.get("seed")
        if reset_seed is not None:
            self.seed_value = int(reset_seed)
            self._map_seed_rng = random.Random(int(reset_seed))
            self._episode_index = 0

        map_seed_opt = options.get("map_seed")
        if map_seed_opt is not None:
            map_seed = int(map_seed_opt)
        else:
            # Avoid accidental repeats even when a caller repeatedly passes reset(seed=...).
            # Mix episode index into RNG stream so each reset gets a distinct map seed.
            stream_seed = self._map_seed_rng.randint(0, 2**31 - 1)
            map_seed = (stream_seed ^ ((self._episode_index + 1) * 1_000_003)) & 0x7FFF_FFFF
            if self._last_map_seed is not None and map_seed == self._last_map_seed:
                map_seed = (map_seed + 1) & 0x7FFF_FFFF

        self._episode_index += 1
        self._last_map_seed = map_seed
        self.env = (
            require_kaggle_env(debug=True, configuration={"randomSeed": int(map_seed)})
            if self.require_kaggle
            else _FakeOrbitWarsBackend(map_seed)
        )
        reset = getattr(self.env, "reset", None)
        if callable(reset):
            reset()
        self.opponent_agent = self._build_opponent_agent()
        self.obs = _extract_player_observation(self.env, self.candidate_player)
        self.turn_index = 0
        self.buffered_actions = []
        self.episode_action_count = 0
        self.episode_invalid_action_count = 0
        self.episode_ships_sent_total = 0.0
        self.episode_target_choice_count = 0
        self.episode_enemy_target_count = 0
        self.episode_neutral_target_count = 0
        self.episode_self_target_count = 0
        self.episode_turn_count = 0
        self.episode_enemy_captures = 0
        self.episode_neutral_captures = 0
        self._rebuild_sources()
        self.previous_total_production = total_production(parse_planets(self.obs), self.candidate_player)
        return self._current_obs(), {"source_id": self._current_source_id(), "no_source": not self.sources, "map_seed": map_seed}

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
        self.episode_action_count += 1
        self.episode_ships_sent_total += float(sum(float(row[2]) for row in decoded_rows if len(row) >= 3))
        action_values = np.asarray(action, dtype=np.float32).reshape(-1).tolist()
        for idx, value in enumerate(action_values[: min(len(chosen), 4)]):
            if int(value) <= 0:
                continue
            self.episode_target_choice_count += 1
            target = chosen[idx]
            if target.owner == (1 - self.candidate_player):
                self.episode_enemy_target_count += 1
            elif target.owner < 0:
                self.episode_neutral_target_count += 1
            elif target.owner == self.candidate_player:
                self.episode_self_target_count += 1
        if any(int(v) > 0 for v in action_values[: min(len(chosen), 4)]) and not decoded_rows:
            self.episode_invalid_action_count += 1
        send_reward = ships_sent_reward(decoded_rows, self.reward_config)
        send_reward += action_targeting_reward(
            source_ships=int(source.ships),
            candidates=chosen,
            action_values=action_values,
            config=self.reward_config,
            reserve_ships=self.action_config.reserve_ships,
        )
        self.buffered_actions.extend(decoded_rows)
        self.current_source_index += 1
        if self.current_source_index < len(self.sources):
            per_action_reward = float((send_reward / max(1, len(self.sources))) * self.reward_config.reward_scale)
            return (
                self._current_obs(),
                per_action_reward,
                False,
                False,
                {
                    "turn_advanced": False,
                    "source_id": self._current_source_id(),
                    "send_reward": send_reward,
                    "send_reward_scaled": per_action_reward,
                },
            )
        return self._advance_turn(self.buffered_actions, send_reward)

    def _advance_turn(self, candidate_actions: list, send_reward: float = 0.0):
        if self.env is None:
            raise RuntimeError("reset() must be called before advancing a turn.")
        if self.opponent_agent is None:
            raise RuntimeError("reset() must build an opponent agent before advancing a turn.")
        opponent_player = 1 - self.candidate_player
        opponent_obs = _extract_player_observation(self.env, opponent_player)
        opponent_actions = self.opponent_agent.act(opponent_obs)
        actions0 = candidate_actions if self.candidate_player == 0 else opponent_actions
        actions1 = opponent_actions if self.candidate_player == 0 else candidate_actions
        production_before = self.previous_total_production
        prod_adv_before = production_advantage(self.obs, self.candidate_player)
        score_adv_before = score_advantage(self.obs, self.candidate_player)
        pressure_before = fleet_pressure_advantage(self.obs, self.candidate_player)
        strategic_before = strategic_score(self.obs, self.candidate_player, self.reward_config)
        ship_score_before = player_score(self.obs, self.candidate_player)
        enemy_ship_score_before = player_score(self.obs, opponent_player)
        ship_adv_before = ship_score_before - enemy_ship_score_before
        planets_before = parse_planets(self.obs)
        _step_kaggle_env(self.env, actions0, actions1)
        self.turn_index += 1
        self.obs = _extract_player_observation(self.env, self.candidate_player)
        planets_after = parse_planets(self.obs)
        current_total = total_production(planets_after, self.candidate_player)
        production_delta = float(current_total - production_before)
        prod_adv_after = production_advantage(self.obs, self.candidate_player)
        score_adv_after = score_advantage(self.obs, self.candidate_player)
        pressure_after = fleet_pressure_advantage(self.obs, self.candidate_player)
        strategic_after = strategic_score(self.obs, self.candidate_player, self.reward_config)
        ship_score_after = player_score(self.obs, self.candidate_player)
        enemy_ship_score_after = player_score(self.obs, opponent_player)
        ship_adv_after = ship_score_after - enemy_ship_score_after
        prod_adv_delta = prod_adv_after - prod_adv_before
        score_adv_delta = score_adv_after - score_adv_before
        strategic_delta = strategic_after - strategic_before
        pressure_delta = self.reward_config.pressure_adv_weight * (pressure_after - pressure_before)
        capture_reward = planet_capture_reward(
            planets_before, planets_after, self.candidate_player, self.reward_config
        )
        for after in planets_after:
            before = next((p for p in planets_before if p.id == after.id), None)
            if before is not None and before.owner != after.owner and after.owner == self.candidate_player:
                if before.owner == opponent_player:
                    self.episode_enemy_captures += 1
                elif before.owner < 0:
                    self.episode_neutral_captures += 1
        passive_penalty = idle_ship_penalty(self.obs, self.candidate_player, self.reward_config)
        invalid_action_penalty = -float(self.episode_invalid_action_count > 0)
        enemy_captured = sum(1 for before in planets_before for after in planets_after if before.id == after.id and before.owner == self.candidate_player and after.owner == opponent_player)
        we_captured = sum(1 for before in planets_before for after in planets_after if before.id == after.id and before.owner != self.candidate_player and after.owner == self.candidate_player)
        net_capture_delta = we_captured - enemy_captured
        ship_delta_reward = self.reward_config.ship_delta_weight * ((ship_score_after - enemy_ship_score_after) / max(1.0, ship_score_after + enemy_ship_score_after))
        production_delta_reward = self.reward_config.production_delta_weight * prod_adv_after
        capture_delta_reward = self.reward_config.net_capture_weight * net_capture_delta
        dense_reward = float(strategic_delta + pressure_delta + ship_delta_reward + production_delta_reward + capture_delta_reward + send_reward)
        clip = self.reward_config.dense_reward_clip
        dense_reward = float(max(-clip, min(clip, dense_reward)))
        num_owned_planets = max(1, sum(1 for p in planets_before if p.owner == self.candidate_player))
        team_reward = float(dense_reward + capture_reward + passive_penalty + invalid_action_penalty)
        reward = float(team_reward / num_owned_planets)
        terminal_reward = 0.0
        self.previous_total_production = current_total
        buffered = list(candidate_actions)
        self.buffered_actions = []
        self._rebuild_sources()
        planets = planets_after
        candidate_has_planets = any(p.owner == self.candidate_player for p in planets)
        opponent_has_planets = any(p.owner == opponent_player for p in planets)
        env_done = _is_kaggle_done(self.env)
        reached_turn_limit = self.turn_index >= self.max_episode_turns
        terminated = env_done or (not candidate_has_planets) or (not opponent_has_planets)
        truncated = reached_turn_limit and not terminated
        if terminated or truncated:
            candidate_score = player_score(self.obs, self.candidate_player)
            opponent_score = player_score(self.obs, opponent_player)
            if truncated and not terminated:
                terminal_reward = timeout_outcome_reward(candidate_score, opponent_score, self.reward_config)
            else:
                terminal_reward = game_outcome_reward(
                    candidate_score=candidate_score,
                    opponent_score=opponent_score,
                    turn_index=self.turn_index,
                    max_episode_turns=self.max_episode_turns,
                    config=self.reward_config,
                )
            reward = float(reward + terminal_reward)
        scaled_reward = float(reward * self.reward_config.reward_scale)
        self.episode_turn_count = self.turn_index
        win_rate = 1.0 if terminated and not truncated and terminal_reward > 0 else 0.0
        loss_rate = 1.0 if terminated and not truncated and terminal_reward < 0 else 0.0
        timeout_rate = 1.0 if truncated else 0.0
        return (
            self._current_obs(),
            scaled_reward,
            bool(terminated),
            bool(truncated),
            {
                "turn_advanced": True,
                "production_before": production_before,
                "production_after": current_total,
                "production_delta": production_delta,
                "capture_reward": capture_reward,
                "send_reward": send_reward,
                "terminal_reward": terminal_reward,
                "buffered_actions": buffered,
                "turn_index": self.turn_index,
                "source_id": self._current_source_id(),
                "production_advantage_before": prod_adv_before,
                "production_advantage": prod_adv_after,
                "production_advantage_delta": prod_adv_delta,
                "score_advantage_before": score_adv_before,
                "score_advantage": score_adv_after,
                "score_advantage_delta": score_adv_delta,
                "pressure_advantage_before": pressure_before,
                "pressure_advantage": pressure_after,
                "strategic_score_before": strategic_before,
                "strategic_score": strategic_after,
                "strategic_score_delta": strategic_delta,
                "pressure_delta": pressure_delta,
                "ship_advantage_before": ship_adv_before,
                "ship_advantage": ship_adv_after,
                "ship_delta_reward": ship_delta_reward,
                "production_delta_reward": production_delta_reward,
                "capture_delta_reward": capture_delta_reward,
                "enemy_planets_captured_us": enemy_captured,
                "our_planets_captured": we_captured,
                "passive_penalty": passive_penalty,
                "dense_reward": dense_reward,
                "team_reward": team_reward,
                "per_action_team_reward": reward,
                "reward_scale": self.reward_config.reward_scale,
                "reward_unscaled_total": reward,
                "reward_total": scaled_reward,
                "reward_scalar_returned_by_step": scaled_reward,
                "episode_components": {
                    "reward/terminal": terminal_reward,
                    "reward/strategic_delta": strategic_delta,
                    "reward/capture": capture_reward,
                    "reward/pressure": pressure_delta,
                    "reward/ship_delta": ship_delta_reward,
                    "reward/production_delta": production_delta_reward,
                    "reward/capture_delta": capture_delta_reward,
                    "reward/local_action": send_reward,
                    "reward/waste_penalty": passive_penalty + invalid_action_penalty,
                    "reward/total": scaled_reward,
                    "game/win_rate": win_rate,
                    "game/loss_rate": loss_rate,
                    "game/timeout_rate": timeout_rate,
                    "game/avg_turns": float(self.episode_turn_count),
                    "game/avg_enemy_captures": float(self.episode_enemy_captures),
                    "game/avg_neutral_captures": float(self.episode_neutral_captures),
                    "game/our_captures_this_turn": float(we_captured),
                    "game/enemy_captures_this_turn": float(enemy_captured),
                    "action/invalid_rate": float(self.episode_invalid_action_count / max(1, self.episode_action_count)),
                    "action/ships_sent_mean": float(self.episode_ships_sent_total / max(1, self.episode_action_count)),
                    "action/enemy_target_rate": float(self.episode_enemy_target_count / max(1, self.episode_target_choice_count)),
                    "action/neutral_target_rate": float(self.episode_neutral_target_count / max(1, self.episode_target_choice_count)),
                    "action/self_target_rate": float(self.episode_self_target_count / max(1, self.episode_target_choice_count)),
                } if (terminated or truncated) else None,
            },
        )

    def _build_opponent_agent(self) -> Any:
        if self.opponent == "starter":
            return StarterAgent()
        loaded_policy = load_any_policy(self.opponent_model)
        return ModelAgent(policy=loaded_policy)

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
