"""Gymnasium planet-step wrapper for PPO Orbit Wars training."""
from __future__ import annotations

from pathlib import Path
import math
import random
from typing import Any

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - exercised only when numpy is missing/broken.
    raise RuntimeError(
        "OrbitWarsPlanetStepEnv requires NumPy. Install dependencies with "
        "`pip install -e .` (or at least `pip install numpy`)."
    ) from exc

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
        class Discrete:
            def __init__(self, n: int):
                self.n = int(n)
                self.shape = ()

    class _Env:
        pass

    class _Gym:
        Env = _Env

    gym = _Gym()  # type: ignore[assignment]
    spaces = _Spaces()  # type: ignore[assignment]

from orbit_wars_rl.agents.heuristic_agent import GreedyAgent, HardAgent
from orbit_wars_rl.agents.model_agent import ModelAgent
from orbit_wars_rl.agents.random_agent import RandomAgent
from orbit_wars_rl.agents.starter_agent import StarterAgent
from orbit_wars_rl.core.actions import (
    ActionDecodeConfig,
    decode_model_outputs,
    get_fleet_speed,
    SEND_FRACTIONS,
)
from orbit_wars_rl.core.candidates import CandidateConfig, comet_ids_from_obs
from orbit_wars_rl.core.geometry import predict_launch, sun_collision_radius_from_obs, trajectory_crosses_sun
from orbit_wars_rl.core.observations import ObservationBuilder
from orbit_wars_rl.core.planets import total_production
from orbit_wars_rl.core.types import Planet, parse_planets, rows
from orbit_wars_rl.env.kaggle_env import require_kaggle_env
from orbit_wars_rl.models.save_load import load_any_policy

MAP_SEEDS = [
    5199, 2083, 3493, 1649, 3233, 405, 3335, 1030, 1467, 78, 32, 1900, 647, 417, 1, 2560,
    272, 585, 1265, 741, 489, 2537, 422, 787, 455, 324, 119, 828, 1049, 906, 1117, 1990,
    5274, 2661, 3774, 2794, 3578, 7045, 4333, 1153, 2412, 1750, 2078, 2957, 1843, 451, 1725, 4676,
    662, 1217, 4461, 4785, 5675, 3403, 4814, 1336, 2996, 2509, 3959, 2867, 2572, 2476, 1282, 2393,
    7542, 6328, 5923, 4252, 4027, 9408, 4693, 2726, 3154, 7166, 6858, 4393, 2177, 2663, 1948, 6475,
    4353, 6462, 5981, 8516, 7770, 6593, 4963, 3473, 3520, 7337, 8763, 9017, 6202, 5047, 1571, 6294,
    8909, 9327, 8514, 9240, 6548, 9658, 9812, 8686, 8717, 8866, 7079, 9306, 7314, 8782, 2419, 8526,
    9118, 9984, 8855, 9227, 8624, 8968, 7186, 9013, 4633, 8603, 9102, 9547, 9453, 7947, 3462, 9427,
]

from orbit_wars_rl.training.reward import (
    RewardShapingConfig,
    game_outcome_reward,
    planet_capture_reward,
    player_score,
    production_advantage,
    score_advantage,
    ships_sent_reward,
    strategic_score,
    timeout_outcome_reward,
    win_speed_bonus,
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


def _terminal_state_rewards(env: Any) -> tuple[float, float] | None:
    """Return final per-player rewards from Kaggle state when available."""
    state = getattr(env, "state", None)
    if not state or len(state) < 2:
        return None
    rewards: list[float] = []
    for slot in state[:2]:
        value = slot.get("reward") if isinstance(slot, dict) else getattr(slot, "reward", None)
        if value is None:
            return None
        rewards.append(float(value))
    return (rewards[0], rewards[1])


class OrbitWarsPlanetStepEnv(gym.Env):
    """Each SB3 step controls one owned planet; full Kaggle turn advances after all sources."""

    metadata = {"render_modes": []}
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
    # action[0]: target selection (0=pass, 1..4=candidate index)
    # action[1]: normalized amount bucket in [0, 100]
    action_space = spaces.MultiDiscrete([5, 101])
    min_source_ships_to_act = 11

    def __init__(
        self,
        opponent: str = "starter",
        opponent_model: str | Path | None = None,
        candidate_player: int = 0,
        seed: int = 0,
        max_episode_turns: int = 400,
        require_kaggle: bool = True,
    ):
        if opponent in {"starter", "random", "greedy", "hard"}:
            if opponent_model is not None:
                raise ValueError("opponent_model must be None unless opponent='model'")
        elif opponent == "model":
            if opponent_model is None:
                raise ValueError("opponent_model is required when opponent='model'")
        else:
            raise ValueError("opponent must be one of: 'starter', 'random', 'greedy', 'hard', 'model'")
        if candidate_player not in (0, 1):
            raise ValueError("candidate_player must be 0 or 1")
        self.opponent = opponent
        self.opponent_model = opponent_model
        self.candidate_player = candidate_player
        self.seed_value = seed
        self._map_seed_rng = random.Random(seed)
        self._episode_index = 0
        self._seed_cycle: list[int] = []
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
        self.episode_return_scaled = 0.0
        self.episode_return_log_scale = 0.1
        self.current_map_seed: int | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        options = options or {}
        reset_seed = seed if seed is not None else options.get("seed")
        if reset_seed is not None and self._episode_index == 0:
            # Some training stacks pass the same reset seed repeatedly.
            # Only apply it on the first reset so later episodes still vary.
            self.seed_value = int(reset_seed)
            self._map_seed_rng = random.Random(int(reset_seed))

        map_seed_opt = options.get("map_seed")
        if map_seed_opt is not None:
            map_seed = int(map_seed_opt)
        else:
            if not self._seed_cycle:
                self._seed_cycle = MAP_SEEDS.copy()
                self._map_seed_rng.shuffle(self._seed_cycle)
            map_seed = self._seed_cycle.pop()

        self.current_map_seed = map_seed
        self._episode_index += 1
        self.env = (
            require_kaggle_env(debug=True, configuration={"seed": int(map_seed)})
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
        self.episode_return_scaled = 0.0
        self._rebuild_sources()
        self.previous_total_production = total_production(parse_planets(self.obs), self.candidate_player)
        return self._current_obs(), {"source_id": self._current_source_id(), "no_source": not self.sources, "map_seed": map_seed}

    def _can_source_send(self, source: Planet, candidates: list[Planet]) -> bool:
        remaining = max(0, int(source.ships) - max(0, self.action_config.reserve_ships))
        return remaining > 0 and bool(candidates)

    def action_masks(self) -> tuple[list[bool], list[bool]]:
        """Mask invalid per-dimension MultiDiscrete choices for the current source."""
        if not self.sources:
            return [True, False, False, False, False], [True] * 101
        source = self.sources[min(self.current_source_index, len(self.sources) - 1)]
        _obs, filtered_candidates, _launches = self._current_obs_and_candidates(source)
        can_send = self._can_source_send(source, filtered_candidates)
        max_target = min(4, len(filtered_candidates))
        target_mask = [True] + [can_send and (i <= max_target) for i in range(1, 5)]
        amount_mask = [True] * 101
        return target_mask, amount_mask

    def step(self, action):
        if self.env is None:
            raise RuntimeError("reset() must be called before step().")
        if not self.sources:
            return self._advance_turn([])
        source = self.sources[self.current_source_index]
        _model_obs, filtered_candidates, proposed_launches = self._current_obs_and_candidates(
            source
        )
        decoded = decode_model_outputs(
            source,
            filtered_candidates,
            np.asarray(action, dtype=np.float32).reshape(-1).tolist(),
            self.action_config,
            angular_velocity=float(self.obs.get("angular_velocity", 0.0)),
            fleet_speed=float(self.obs.get("fleet_speed", 1.0)),
            sun_radius=sun_collision_radius_from_obs(self.obs),
            precomputed_launches=proposed_launches,
        )
        decoded_rows = rows(decoded)
        self.episode_action_count += 1
        self.episode_ships_sent_total += float(sum(float(row[2]) for row in decoded_rows if len(row) >= 3))
        action_values = np.asarray(action, dtype=np.float32).reshape(-1).tolist()
        target_choice = int(action_values[0]) if action_values else 0
        if 1 <= target_choice <= min(len(filtered_candidates), 4):
            self.episode_target_choice_count += 1
            target = filtered_candidates[target_choice - 1]
            if target.owner == (1 - self.candidate_player):
                self.episode_enemy_target_count += 1
            elif target.owner < 0:
                self.episode_neutral_target_count += 1
            elif target.owner == self.candidate_player:
                self.episode_self_target_count += 1
        chose_send = target_choice > 0
        can_send = self._can_source_send(source, filtered_candidates)
        if chose_send and can_send and not decoded_rows:
            self.episode_invalid_action_count += 1
        tactical_reward = self._source_tactical_reward(source, filtered_candidates, decoded_rows, action_values)
        send_reward = ships_sent_reward(decoded_rows, self.reward_config) + tactical_reward
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
        strategic_after = strategic_score(self.obs, self.candidate_player, self.reward_config)
        ship_score_after = player_score(self.obs, self.candidate_player)
        enemy_ship_score_after = player_score(self.obs, opponent_player)
        ship_adv_after = ship_score_after - enemy_ship_score_after
        prod_adv_delta = prod_adv_after - prod_adv_before
        score_adv_delta = score_adv_after - score_adv_before
        strategic_delta = strategic_after - strategic_before
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
        invalid_action_penalty = 0.0
        enemy_captured = sum(1 for before in planets_before for after in planets_after if before.id == after.id and before.owner == self.candidate_player and after.owner == opponent_player)
        we_captured = sum(1 for before in planets_before for after in planets_after if before.id == after.id and before.owner != self.candidate_player and after.owner == self.candidate_player)
        net_capture_delta = we_captured - enemy_captured
        ship_delta_reward = self.reward_config.ship_delta_weight * ((ship_score_after - enemy_ship_score_after) / max(1.0, ship_score_after + enemy_ship_score_after))
        production_delta_reward = self.reward_config.production_delta_weight * prod_adv_after
        capture_delta_reward = self.reward_config.net_capture_weight * net_capture_delta
        dense_reward = float(strategic_delta + ship_delta_reward + production_delta_reward + capture_delta_reward + send_reward)
        clip = self.reward_config.dense_reward_clip
        dense_reward = float(max(-clip, min(clip, dense_reward)))
        num_owned_planets = max(1, sum(1 for p in planets_before if p.owner == self.candidate_player))
        team_reward = float(dense_reward + capture_reward + invalid_action_penalty)
        reward = float(team_reward / num_owned_planets)
        terminal_reward = 0.0
        state_rewards: tuple[float, float] | None = None
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
            state_rewards = _terminal_state_rewards(self.env)
            if truncated and not terminated:
                terminal_reward = timeout_outcome_reward(candidate_score, opponent_score, self.reward_config)
            else:
                if state_rewards is not None and state_rewards[0] != state_rewards[1]:
                    candidate_state_reward = state_rewards[self.candidate_player]
                    opponent_state_reward = state_rewards[opponent_player]
                    if candidate_state_reward > opponent_state_reward:
                        terminal_reward = float(
                            self.reward_config.win_reward
                            + win_speed_bonus(self.turn_index, self.max_episode_turns, self.reward_config)
                        )
                    else:
                        progress = min(max(self.turn_index, 1), self.max_episode_turns) / max(1, self.max_episode_turns)
                        terminal_reward = float(self.reward_config.loss_penalty + self.reward_config.loss_survival_bonus * progress)
                else:
                    terminal_reward = game_outcome_reward(
                        candidate_score=candidate_score,
                        opponent_score=opponent_score,
                        turn_index=self.turn_index,
                        max_episode_turns=self.max_episode_turns,
                        config=self.reward_config,
                    )
            if (
                self.current_map_seed is not None
                and candidate_score < opponent_score
                and self.current_map_seed not in self._seed_cycle
            ):
                self._seed_cycle.insert(0, self.current_map_seed)
            reward = float(reward + terminal_reward)
        scaled_reward = float(reward * self.reward_config.reward_scale)
        self.episode_return_scaled += scaled_reward * self.episode_return_log_scale
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
                "terminal_state_rewards": state_rewards,
                "buffered_actions": buffered,
                "turn_index": self.turn_index,
                "source_id": self._current_source_id(),
                "production_advantage_before": prod_adv_before,
                "production_advantage": prod_adv_after,
                "production_advantage_delta": prod_adv_delta,
                "score_advantage_before": score_adv_before,
                "score_advantage": score_adv_after,
                "score_advantage_delta": score_adv_delta,
                "strategic_score_before": strategic_before,
                "strategic_score": strategic_after,
                "strategic_score_delta": strategic_delta,
                "ship_advantage_before": ship_adv_before,
                "ship_advantage": ship_adv_after,
                "ship_delta_reward": ship_delta_reward,
                "production_delta_reward": production_delta_reward,
                "capture_delta_reward": capture_delta_reward,
                "enemy_planets_captured_us": enemy_captured,
                "our_planets_captured": we_captured,
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
                    "reward/ship_delta": ship_delta_reward,
                    "reward/production_delta": production_delta_reward,
                    "reward/capture_delta": capture_delta_reward,
                    "reward/local_action": send_reward,
                    "reward/total": scaled_reward,
                    "reward/episode_return": float(self.episode_return_scaled),
                    "train_rollout/stochastic_win_rate": win_rate,
                    "game/loss_rate": loss_rate,
                    "game/timeout_rate": timeout_rate,
                    "game/avg_turns": float(self.episode_turn_count),
                    "game/avg_enemy_captures": float(self.episode_enemy_captures),
                    "game/avg_neutral_captures": float(self.episode_neutral_captures),
                    "game/map_seed": float(self.current_map_seed or -1),
                    "game/our_captures_this_turn": float(we_captured),
                    "game/enemy_captures_this_turn": float(enemy_captured),
                    "action/invalid_rate": float(self.episode_invalid_action_count / max(1, self.episode_action_count)),
                    "action/ships_sent_mean": float(self.episode_ships_sent_total / max(1, self.episode_action_count)),
                    "game/pool_size": float(len(self._seed_cycle)),
                } if (terminated or truncated) else None,
            },
        )

    def _build_opponent_agent(self) -> Any:
        if self.opponent == "starter":
            return StarterAgent()
        if self.opponent == "random":
            return RandomAgent(seed=self.seed_value + self._episode_index)
        if self.opponent == "greedy":
            return GreedyAgent()
        if self.opponent == "hard":
            return HardAgent()
        loaded_policy = load_any_policy(self.opponent_model)
        return ModelAgent(policy=loaded_policy)

    def _rebuild_sources(self) -> None:
        planets = parse_planets(self.obs)
        comet_ids = comet_ids_from_obs(self.obs)
        self.sources = [
            p
            for p in planets
            if p.owner == self.candidate_player
            and p.id not in comet_ids
            and int(p.ships) >= self.min_source_ships_to_act
        ]
        self.current_source_index = 0

    def _source_tactical_reward(
        self, source: Planet, candidates: list[Planet], decoded_rows: list[list[float]], action_values: list[float]
    ) -> float:
        if not decoded_rows:
            return 0.01
        local_reward = 0.0
        remaining = max(0, int(source.ships) - max(0, self.action_config.reserve_ships))
        output_len = 8 if len(action_values) == 9 else len(action_values)
        sun_radius = sun_collision_radius_from_obs(self.obs)
        angular_velocity = float(self.obs.get("angular_velocity", 0.0))
        if output_len == 2:
            target_choice = int(action_values[0]) if action_values else 0
            if target_choice <= 0:
                return local_reward
            idx = target_choice - 1
            if idx < 0 or idx >= min(4, len(candidates)):
                return local_reward
            amount_value = float(action_values[1]) if len(action_values) > 1 else 0.0
            pct = max(0.0, min(1.0, amount_value / 100.0 if amount_value > 1.0 else amount_value))
            min_send = min(10, remaining)
            span = max(0, remaining - min_send)
            ships = int(min_send + math.floor(pct * span))
            ships = min(max(min_send, ships), remaining)
            if ships <= 0:
                return local_reward
            target = candidates[idx]
            launch = predict_launch(source, target, angular_velocity, get_fleet_speed(ships))
            if trajectory_crosses_sun(launch.source_xy, launch.target_xy, sun_radius=sun_radius):
                return local_reward
            if target.owner != self.candidate_player and ships < int(target.ships):
                target_ships = max(1, int(target.ships))
                shortfall = target_ships - ships
                normalizer = max(1, target_ships - 1)
                local_reward -= 0.1 * (shortfall / normalizer)
            return local_reward

        for idx in range(min(4, len(candidates))):
            if remaining <= 0:
                break
            if output_len == 4:
                discrete = max(0, min(5, int(action_values[idx])))
                requested = int(remaining * SEND_FRACTIONS[discrete])
            else:
                if (idx * 2 + 1) >= len(action_values):
                    break
                active = float(action_values[idx * 2]) > self.action_config.activation_threshold
                if not active:
                    continue
                pct = max(0.0, min(1.0, float(action_values[idx * 2 + 1])))
                requested = int(source.ships * pct)
                requested = max(1, requested)
            ships = min(max(0, requested), remaining)
            if ships <= 0:
                continue
            target = candidates[idx]
            launch = predict_launch(source, target, angular_velocity, get_fleet_speed(ships))
            if trajectory_crosses_sun(launch.source_xy, launch.target_xy, sun_radius=sun_radius):
                continue
            remaining -= ships
            if target.owner != self.candidate_player and ships < int(target.ships):
                target_ships = max(1, int(target.ships))
                shortfall = target_ships - ships
                normalizer = max(1, target_ships - 1)
                local_reward -= 0.1 * (shortfall / normalizer)
        return local_reward

    def _current_source_id(self) -> int | None:
        if not self.sources:
            return None
        return self.sources[min(self.current_source_index, len(self.sources) - 1)].id

    def _current_obs(self) -> np.ndarray:
        if not self.sources:
            return np.zeros((15,), dtype=np.float32)
        source = self.sources[min(self.current_source_index, len(self.sources) - 1)]
        vec, _chosen, _launches = self._current_obs_and_candidates(source)
        return vec

    def _current_obs_and_candidates(
        self, source: Planet
    ) -> tuple[np.ndarray, list[Planet], list]:
        vec, chosen, launches = self.builder.build_filtered_for_source(
            source,
            parse_planets(self.obs),
            self.candidate_player,
            previous_total_production=self.previous_total_production,
            comet_ids=comet_ids_from_obs(self.obs),
            angular_velocity=float(self.obs.get("angular_velocity", 0.0)),
            fleet_speed=float(self.obs.get("fleet_speed", 1.0)),
            sun_radius=sun_collision_radius_from_obs(self.obs),
        )
        return np.asarray(list(vec), dtype=np.float32), chosen, launches
