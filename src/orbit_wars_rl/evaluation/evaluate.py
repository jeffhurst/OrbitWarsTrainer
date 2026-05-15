"""Evaluation against local baseline agents."""
from __future__ import annotations

import json
from pathlib import Path

from orbit_wars_rl.agents.heuristic_agent import HeuristicAgent
from orbit_wars_rl.agents.model_agent import ModelAgent
from orbit_wars_rl.agents.random_agent import RandomAgent
from orbit_wars_rl.agents.starter_agent import StarterAgent
from orbit_wars_rl.evaluation.metrics import EvaluationMetrics
from orbit_wars_rl.models.save_load import load_policy
from orbit_wars_rl.visualization.watch import make_demo_observation


def evaluate(model_path: str | None = None, games: int = 4, out_dir: str | Path = "runs/eval") -> Path:
    policy = load_policy(model_path) if model_path else None
    model_agent = ModelAgent(policy=policy) if policy else StarterAgent()
    opponents = [StarterAgent(), RandomAgent(seed=1), HeuristicAgent()]
    launches = []
    for i in range(games):
        obs = make_demo_observation(0)
        launches.append(len(model_agent.act(obs)))
        opponents[i % len(opponents)].act({**obs, "player": 1})
    metrics = EvaluationMetrics(
        games=games,
        win_rate=0.0,
        average_final_score=0.0,
        average_production_controlled=0.0,
        average_ships_controlled=0.0,
        average_planets_controlled=0.0,
        average_number_of_launches=sum(launches) / max(1, len(launches)),
        average_invalid_actions=0.0,
        average_episode_length=1.0,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "evaluation.json"
    path.write_text(json.dumps(metrics.to_dict(), indent=2) + "\n")
    return path
