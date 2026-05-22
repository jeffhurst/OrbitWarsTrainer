#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbit_wars_rl.agents.model_agent import ModelAgent
from orbit_wars_rl.models.save_load import load_policy
from orbit_wars_rl.visualization.watch import watch_agents


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and render a real Kaggle model-vs-model Orbit Wars replay.")
    parser.add_argument("--model", required=True, help="Path to the player 0 model/policy file.")
    parser.add_argument("--opponent", required=True, help="Path to the player 1 model/policy file.")
    parser.add_argument("--out-dir", default="runs/watch", help="Directory for replay artifacts.")
    parser.add_argument("--name", default="model_vs_model", help="Replay artifact basename.")
    parser.add_argument("--seed", type=int, default=None, help="Kaggle map seed to force a specific map layout.")
    args = parser.parse_args()

    agent = ModelAgent(policy=load_policy(args.model))
    opponent = ModelAgent(policy=load_policy(args.opponent))
    paths = watch_agents(agent.act, opponent.act, out_dir=args.out_dir, name=args.name, seed=args.seed)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
