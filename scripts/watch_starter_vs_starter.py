#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbit_wars_rl.agents.starter_agent import StarterAgent
from orbit_wars_rl.visualization.watch import watch_agents


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and render a real Kaggle starter-vs-starter Orbit Wars replay.")
    parser.add_argument("--out-dir", default="runs/watch", help="Directory for replay artifacts.")
    parser.add_argument("--name", default="starter_vs_starter", help="Replay artifact basename.")
    args = parser.parse_args()

    paths = watch_agents(StarterAgent().act, StarterAgent().act, out_dir=args.out_dir, name=args.name)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
