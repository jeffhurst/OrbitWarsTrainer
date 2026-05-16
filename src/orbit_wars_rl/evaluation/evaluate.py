"""Evaluation helpers for bootstrap and SB3 PPO policies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orbit_wars_rl.agents.model_agent import ModelAgent
from orbit_wars_rl.agents.starter_agent import StarterAgent
from orbit_wars_rl.core.planets import total_production
from orbit_wars_rl.core.types import parse_planets
from orbit_wars_rl.env.ppo_planet_env import OrbitWarsPlanetStepEnv, _extract_player_observation
from orbit_wars_rl.models.save_load import load_policy
from orbit_wars_rl.visualization.watch import kaggle_agent
from orbit_wars_rl.env.kaggle_env import require_kaggle_env


def _write_metrics(metrics: dict[str, Any], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "evaluation.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    return path


def _evaluate_fake(policy, games: int, out_dir: str | Path) -> Path:
    rewards: list[float] = []
    prod_deltas: list[float] = []
    final_candidate: list[float] = []
    final_opponent: list[float] = []
    lengths: list[int] = []
    for game in range(games):
        env = OrbitWarsPlanetStepEnv(require_kaggle=False, seed=game, max_episode_turns=25)
        obs, _info = env.reset(seed=game)
        done = False
        total_reward = 0.0
        length = 0
        while not done:
            action = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if info.get("turn_advanced"):
                prod_deltas.append(float(info.get("production_delta", 0.0)))
                length += 1
            done = bool(terminated or truncated)
        planets = parse_planets(env.obs)
        rewards.append(total_reward)
        final_candidate.append(total_production(planets, 0))
        final_opponent.append(total_production(planets, 1))
        lengths.append(length)
    metrics = {
        "games": games,
        "average_reward": sum(rewards) / max(1, len(rewards)),
        "average_production_delta": sum(prod_deltas) / max(1, len(prod_deltas)),
        "final_average_candidate_production": sum(final_candidate) / max(1, len(final_candidate)),
        "final_average_opponent_production": sum(final_opponent) / max(1, len(final_opponent)),
        "average_episode_length": sum(lengths) / max(1, len(lengths)),
        "win_rate": sum(c > o for c, o in zip(final_candidate, final_opponent)) / max(1, games),
        "invalid_actions": 0.0,
        "backend": "fake-smoke",
    }
    return _write_metrics(metrics, out_dir)


def _evaluate_kaggle(policy, games: int, out_dir: str | Path) -> Path:
    rewards: list[float] = []
    final_candidate: list[float] = []
    final_opponent: list[float] = []
    lengths: list[int] = []
    wins = 0
    for _game in range(games):
        env = require_kaggle_env(debug=True)
        candidate = ModelAgent(policy=policy)
        opponent = StarterAgent()
        env.run([kaggle_agent(candidate.act), kaggle_agent(opponent.act)])
        try:
            obs0 = _extract_player_observation(env, 0)
            obs1 = _extract_player_observation(env, 1)
            p0 = total_production(parse_planets(obs0), 0)
            p1 = total_production(parse_planets(obs1), 1)
        except Exception:
            p0 = p1 = 0.0
        final_candidate.append(p0)
        final_opponent.append(p1)
        rewards.append(p0)
        wins += int(p0 > p1)
        steps = None
        try:
            data = env.toJSON()
            steps = len(data.get("steps", [])) if isinstance(data, dict) else None
        except Exception:
            pass
        lengths.append(int(steps or 0))
    metrics = {
        "games": games,
        "average_reward": sum(rewards) / max(1, len(rewards)),
        "average_production_delta": 0.0,
        "final_average_candidate_production": sum(final_candidate) / max(1, len(final_candidate)),
        "final_average_opponent_production": sum(final_opponent) / max(1, len(final_opponent)),
        "average_episode_length": sum(lengths) / max(1, len(lengths)),
        "win_rate": wins / max(1, games),
        "invalid_actions": 0.0,
        "backend": "kaggle",
    }
    return _write_metrics(metrics, out_dir)


def evaluate(
    model_path: str | None = None,
    games: int = 4,
    out_dir: str | Path = "runs/eval",
    require_kaggle: bool = True,
) -> Path:
    policy = load_policy(model_path)
    if require_kaggle:
        return _evaluate_kaggle(policy, games, out_dir)
    return _evaluate_fake(policy, games, out_dir)
