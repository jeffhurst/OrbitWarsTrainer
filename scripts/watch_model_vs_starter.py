#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
from orbit_wars_rl.agents.model_agent import ModelAgent
from orbit_wars_rl.agents.starter_agent import StarterAgent
from orbit_wars_rl.models.save_load import load_policy
from orbit_wars_rl.visualization.watch import watch_agents

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    agent = ModelAgent(policy=load_policy(args.model)) if args.model else ModelAgent()
    paths = watch_agents(agent.act, StarterAgent().act, out_dir="runs/watch", steps=3)
    for path in paths:
        print(f"wrote {path}")
