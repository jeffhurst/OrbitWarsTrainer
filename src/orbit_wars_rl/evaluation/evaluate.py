"""Evaluation helpers for bootstrap and SB3 PPO policies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orbit_wars_rl.core.planets import total_production
from orbit_wars_rl.core.types import parse_planets
from orbit_wars_rl.env.ppo_planet_env import OrbitWarsPlanetStepEnv
from orbit_wars_rl.models.save_load import load_policy
from orbit_wars_rl.training.reward import player_score


def _write_metrics(metrics: dict[str, Any], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "evaluation.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    return path


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _evaluate_planet_step(
    policy,
    games: int,
    out_dir: str | Path,
    *,
    require_kaggle: bool,
    backend: str,
    max_episode_turns: int,
) -> Path:
    """Evaluate by stepping the same planet-step environment used for training.

    ``average_reward`` is the mean episodic sum of the shaped training reward emitted by
    :class:`OrbitWarsPlanetStepEnv` for both fake and Kaggle-backed evaluation. The win rate uses
    the same terminal outcome score as training: total ships on planets and in flight.
    """
    rewards: list[float] = []
    prod_deltas: list[float] = []
    final_candidate_production: list[float] = []
    final_opponent_production: list[float] = []
    final_candidate_scores: list[float] = []
    final_opponent_scores: list[float] = []
    lengths: list[int] = []
    for game in range(games):
        env = OrbitWarsPlanetStepEnv(
            require_kaggle=require_kaggle,
            seed=game,
            max_episode_turns=max_episode_turns,
        )
        obs, _info = env.reset(seed=game)
        done = False
        total_reward = 0.0
        length = 0
        while not done:
            action = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if info.get("turn_advanced"):
                prod_deltas.append(float(info.get("production_delta", 0.0)))
                length += 1
            done = bool(terminated or truncated)
        planets = parse_planets(env.obs)
        rewards.append(total_reward)
        final_candidate_production.append(total_production(planets, 0))
        final_opponent_production.append(total_production(planets, 1))
        final_candidate_scores.append(player_score(env.obs, 0))
        final_opponent_scores.append(player_score(env.obs, 1))
        lengths.append(length)
    metrics = {
        "games": games,
        "average_reward": _mean(rewards),
        "average_production_delta": _mean(prod_deltas),
        "final_average_candidate_production": _mean(final_candidate_production),
        "final_average_opponent_production": _mean(final_opponent_production),
        "final_average_candidate_score": _mean(final_candidate_scores),
        "final_average_opponent_score": _mean(final_opponent_scores),
        "average_episode_length": _mean([float(length) for length in lengths]),
        "win_rate": sum(c > o for c, o in zip(final_candidate_scores, final_opponent_scores))
        / max(1, games),
        "invalid_actions": 0.0,
        "backend": backend,
    }
    return _write_metrics(metrics, out_dir)


def _evaluate_fake(policy, games: int, out_dir: str | Path) -> Path:
    return _evaluate_planet_step(
        policy,
        games,
        out_dir,
        require_kaggle=False,
        backend="fake-smoke",
        max_episode_turns=25,
    )


def _evaluate_kaggle(policy, games: int, out_dir: str | Path) -> Path:
    return _evaluate_planet_step(
        policy,
        games,
        out_dir,
        require_kaggle=True,
        backend="kaggle",
        max_episode_turns=400,
    )


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
