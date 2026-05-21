#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbit_wars_rl.agents.heuristic_agent import GreedyAgent, HardAgent
from orbit_wars_rl.agents.model_agent import ModelAgent
from orbit_wars_rl.agents.random_agent import RandomAgent
from orbit_wars_rl.agents.starter_agent import StarterAgent
from orbit_wars_rl.models.save_load import load_policy
from orbit_wars_rl.visualization.watch import watch_agents


def make_opponent(name: str):
    mapping = {
        "random": RandomAgent,
        "greedy": GreedyAgent,
        "hard": HardAgent,
        "starter": StarterAgent,
    }
    try:
        return mapping[name]()
    except KeyError as exc:
        valid = ", ".join(mapping)
        raise ValueError(f"Unknown opponent '{name}'. Choose one of: {valid}.") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and render a model-vs-scripted-agent Orbit Wars replay.")
    parser.add_argument("--model", required=True, help="Path to model zip (e.g. runs/models/model_to_watch.zip).")
    parser.add_argument(
        "--opponent",
        required=True,
        choices=["random", "greedy", "hard", "starter"],
        help="Scripted opponent agent name.",
    )
    parser.add_argument("--out-dir", default="runs/watch", help="Directory for replay artifacts.")
    parser.add_argument("--name", default=None, help="Replay artifact basename.")
    parser.add_argument("--seed", type=int, default=None, help="Kaggle map seed to force a specific map layout.")
    args = parser.parse_args()

    model_agent = ModelAgent(policy=load_policy(args.model))
    opponent_name = args.opponent.lower()
    opponent_agent = make_opponent(opponent_name)
    replay_name = args.name or f"model_vs_{opponent_name}"

    paths = watch_agents(model_agent.act, opponent_agent.act, out_dir=args.out_dir, name=replay_name, seed=args.seed)
    for path in paths:
        print(f"wrote {path}")
